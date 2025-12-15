"""
Behavioral Cloning - Learn from expert demonstrations.
Much faster than RL for basic locomotion.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class DemonstrationDataset(Dataset):
    """Dataset of expert demonstrations."""
    
    def __init__(self, demonstrations):
        self.data = []
        
        for episode in demonstrations:
            for transition in episode:
                # Input: observation + target position
                obs = transition['obs']  # 13 values
                target = np.array([transition['target_x'], transition['target_y']])  # 2 values
                input_data = np.concatenate([obs, target])  # 15 values total
                
                # Output: action
                action = transition['action']  # 4 values
                
                self.data.append((input_data, action))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data, action = self.data[idx]
        return torch.FloatTensor(input_data), torch.FloatTensor(action)


class BehavioralCloningAgent(nn.Module):
    """Neural network that imitates expert behavior."""
    
    def __init__(self, input_dim=15, output_dim=4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_behavioral_cloning(demonstrations_file='expert_demonstrations.pkl', 
                           epochs=200, batch_size=64, lr=1e-3):
    """Train behavioral cloning agent."""
    
    # Load demonstrations
    print("Loading demonstrations...")
    with open(demonstrations_file, 'rb') as f:
        demonstrations = pickle.load(f)
    
    print(f"Loaded {len(demonstrations)} episodes")
    
    # Create dataset
    dataset = DemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} transitions")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehavioralCloningAgent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Training on {device}")
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_input, batch_action in dataloader:
            batch_input = batch_input.to(device)
            batch_action = batch_action.to(device)
            
            # Forward pass
            predicted_action = model(batch_input)
            loss = criterion(predicted_action, batch_action)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    model_path = 'behavioral_cloning_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Behavioral Cloning Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_curve.png')
    print("Training curve saved to training_curve.png")
    
    return model


def test_behavioral_cloning(model_path='behavioral_cloning_model.pth', 
                          num_episodes=10, render=False, difficulty='hard'):
    """Test the behavioral cloning agent."""
    from robot_basic_env import RobotBasicEnv
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehavioralCloningAgent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Test environment with difficulty setting
    env = RobotBasicEnv()
    
    # Set difficulty level
    if difficulty == 'easy':
        target_radius = 1.0
        time_pressure = False
        print(f"Testing with EASY difficulty: target radius {target_radius}m")
    elif difficulty == 'medium':
        target_radius = 0.7
        time_pressure = False
        print(f"Testing with MEDIUM difficulty: target radius {target_radius}m")
    elif difficulty == 'hard':
        target_radius = 0.4
        time_pressure = True
        print(f"Testing with HARD difficulty: target radius {target_radius}m, time pressure ON")
    else:
        target_radius = 1.0
        time_pressure = False
        print(f"Testing with DEFAULT difficulty: target radius {target_radius}m")
    
    env.set_difficulty(target_radius=target_radius, time_pressure=time_pressure)
    
    episode_returns = []
    targets_reached = []
    
    print(f"Testing behavioral cloning agent...")
    
    if render:
        # Use MuJoCo viewer for rendering
        import mujoco
        from mujoco import viewer
        
        m = env.model
        d = env.data
        
        robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        
        with viewer.launch_passive(m, d) as v:
            cam = v.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = robot_body_id
            cam.azimuth = 180
            cam.elevation = -20
            cam.distance = 15
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                done = False
                episode_return = 0
                targets_hit = 0
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"Target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                
                while not done and v.is_running():
                    # Prepare input
                    target = np.array([info['target_x'], info['target_y']])
                    input_data = np.concatenate([obs, target])
                    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
                    
                    # Get action
                    with torch.no_grad():
                        action = model(input_tensor).cpu().numpy()[0]
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_return += reward
                    
                    # Count targets
                    if reward > 50:  # Target reached
                        targets_hit += 1
                        print(f"  Target reached! New target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                    
                    # Render
                    cam.trackbodyid = robot_body_id
                    v.sync()
                    import time
                    time.sleep(0.01)  # 100Hz
                
                episode_returns.append(episode_return)
                targets_reached.append(targets_hit)
                
                print(f"  Return: {episode_return:.2f}, Targets: {targets_hit}")
                
                if not v.is_running():
                    break
    else:
        # No rendering - just run episodes
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0
            targets_hit = 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            
            while not done:
                # Prepare input
                target = np.array([info['target_x'], info['target_y']])
                input_data = np.concatenate([obs, target])
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
                
                # Get action
                with torch.no_grad():
                    action = model(input_tensor).cpu().numpy()[0]
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                
                # Count targets
                if reward > 50:  # Target reached
                    targets_hit += 1
            
            episode_returns.append(episode_return)
            targets_reached.append(targets_hit)
            
            print(f"  Return: {episode_return:.2f}, Targets: {targets_hit}")
    
    env.close()
    
    print(f"\nResults:")
    print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average Targets: {np.mean(targets_reached):.2f}")
    print(f"Success Rate: {100 * sum(t > 0 for t in targets_reached) / num_episodes:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train behavioral cloning")
    parser.add_argument("--test", action="store_true", help="Test behavioral cloning")
    parser.add_argument("--render", action="store_true", help="Render during testing")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"], 
                       help="Test difficulty level")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    args = parser.parse_args()
    
    if args.train:
        train_behavioral_cloning(epochs=args.epochs)
    
    if args.test:
        test_behavioral_cloning(render=args.render, difficulty=args.difficulty)