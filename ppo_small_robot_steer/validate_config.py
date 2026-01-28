#!/usr/bin/env python3
"""
Script pour valider et tester les fichiers de configuration JSON.
"""
import json
import os
import sys
from typing import Dict, Any, List


def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """Valide la structure du fichier de configuration."""
    errors = []
    
    # Sections requises
    required_sections = ['training', 'ppo', 'environment', 'network']
    for section in required_sections:
        if section not in config:
            errors.append(f"Section manquante: {section}")
    
    # Validation section training
    if 'training' in config:
        training = config['training']
        required_training = ['total_timesteps', 'num_envs', 'num_steps', 'seed']
        for param in required_training:
            if param not in training:
                errors.append(f"Paramètre manquant dans training: {param}")
            elif not isinstance(training[param], int):
                errors.append(f"training.{param} doit être un entier")
    
    # Validation section ppo
    if 'ppo' in config:
        ppo = config['ppo']
        required_ppo = ['lr', 'gamma', 'gae_lambda', 'clip_coef']
        for param in required_ppo:
            if param not in ppo:
                errors.append(f"Paramètre manquant dans ppo: {param}")
            elif not isinstance(ppo[param], (int, float)):
                errors.append(f"ppo.{param} doit être un nombre")
    
    # Validation section environment
    if 'environment' in config:
        env = config['environment']
        if 'max_steps' in env and not isinstance(env['max_steps'], int):
            errors.append("environment.max_steps doit être un entier")
    
    # Validation section network
    if 'network' in config:
        network = config['network']
        list_params = ['robot_net_hidden', 'history_net_hidden', 'cnn_channels', 'backbone_hidden']
        for param in list_params:
            if param in network and not isinstance(network[param], list):
                errors.append(f"network.{param} doit être une liste")
    
    return errors


def validate_config_values(config: Dict[str, Any]) -> List[str]:
    """Valide les valeurs des paramètres."""
    warnings = []
    
    # Vérifications training
    if 'training' in config:
        training = config['training']
        
        if training.get('num_envs', 0) < 1:
            warnings.append("num_envs devrait être >= 1")
        
        if training.get('num_steps', 0) < 1:
            warnings.append("num_steps devrait être >= 1")
        
        if training.get('total_timesteps', 0) < 1000:
            warnings.append("total_timesteps semble très faible")
    
    # Vérifications ppo
    if 'ppo' in config:
        ppo = config['ppo']
        
        if ppo.get('lr', 0) <= 0 or ppo.get('lr', 0) > 1:
            warnings.append("lr devrait être entre 0 et 1")
        
        if ppo.get('gamma', 0) <= 0 or ppo.get('gamma', 0) > 1:
            warnings.append("gamma devrait être entre 0 et 1")
        
        if ppo.get('gae_lambda', 0) <= 0 or ppo.get('gae_lambda', 0) > 1:
            warnings.append("gae_lambda devrait être entre 0 et 1")
        
        if ppo.get('clip_coef', 0) <= 0 or ppo.get('clip_coef', 0) > 1:
            warnings.append("clip_coef devrait être entre 0 et 1")
    
    # Vérifications environment
    if 'environment' in config:
        env = config['environment']
        
        if env.get('max_steps', 0) < 100:
            warnings.append("max_steps semble très faible (< 100)")
        
        if env.get('max_steps', 0) > 10000:
            warnings.append("max_steps semble très élevé (> 10000)")
    
    return warnings


def load_and_validate_config(config_path: str) -> bool:
    """Charge et valide un fichier de configuration."""
    print(f"🔍 Validation de {config_path}")
    print("=" * 50)
    
    # Vérifier existence du fichier
    if not os.path.exists(config_path):
        print(f"❌ Fichier non trouvé: {config_path}")
        return False
    
    # Charger le JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ JSON valide")
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur de lecture: {e}")
        return False
    
    # Valider la structure
    errors = validate_config_structure(config)
    if errors:
        print("\n❌ ERREURS DE STRUCTURE:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Structure valide")
    
    # Valider les valeurs
    warnings = validate_config_values(config)
    if warnings:
        print("\n⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Valeurs cohérentes")
    
    # Afficher résumé
    print("\n📊 RÉSUMÉ DE LA CONFIGURATION:")
    if 'training' in config:
        training = config['training']
        print(f"  Timesteps: {training.get('total_timesteps', 'N/A'):,}")
        print(f"  Environnements: {training.get('num_envs', 'N/A')}")
        print(f"  Steps/rollout: {training.get('num_steps', 'N/A')}")
        
        if 'num_envs' in training and 'num_steps' in training:
            batch_size = training['num_envs'] * training['num_steps']
            print(f"  Batch size: {batch_size:,}")
    
    if 'ppo' in config:
        ppo = config['ppo']
        print(f"  Learning rate: {ppo.get('lr', 'N/A')}")
        print(f"  Gamma: {ppo.get('gamma', 'N/A')}")
    
    if 'environment' in config:
        env = config['environment']
        print(f"  Max steps/épisode: {env.get('max_steps', 'N/A')}")
        corridor = env.get('corridor_xml', None)
        if corridor:
            print(f"  Corridor: {corridor}")
        else:
            print(f"  Corridor: aléatoire")
    
    print("=" * 50)
    return len(errors) == 0


def compare_configs(config1_path: str, config2_path: str):
    """Compare deux fichiers de configuration."""
    print(f"🔄 Comparaison: {config1_path} vs {config2_path}")
    print("=" * 60)
    
    try:
        with open(config1_path, 'r') as f:
            config1 = json.load(f)
        with open(config2_path, 'r') as f:
            config2 = json.load(f)
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return
    
    def compare_section(section_name: str, dict1: dict, dict2: dict, prefix=""):
        differences = []
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in sorted(all_keys):
            full_key = f"{prefix}{section_name}.{key}" if prefix else f"{section_name}.{key}"
            
            if key not in dict1:
                differences.append(f"  + {full_key}: {dict2[key]} (nouveau)")
            elif key not in dict2:
                differences.append(f"  - {full_key}: {dict1[key]} (supprimé)")
            elif dict1[key] != dict2[key]:
                differences.append(f"  ~ {full_key}: {dict1[key]} → {dict2[key]}")
        
        return differences
    
    all_differences = []
    all_sections = set(config1.keys()) | set(config2.keys())
    
    for section in sorted(all_sections):
        if section.startswith('_'):  # Ignorer les commentaires
            continue
            
        if section not in config1:
            all_differences.append(f"+ Section {section}: nouvelle")
        elif section not in config2:
            all_differences.append(f"- Section {section}: supprimée")
        elif isinstance(config1[section], dict) and isinstance(config2[section], dict):
            section_diffs = compare_section(section, config1[section], config2[section])
            all_differences.extend(section_diffs)
        elif config1[section] != config2[section]:
            all_differences.append(f"~ {section}: {config1[section]} → {config2[section]}")
    
    if all_differences:
        print("📋 DIFFÉRENCES TROUVÉES:")
        for diff in all_differences:
            print(diff)
    else:
        print("✅ Configurations identiques")
    
    print("=" * 60)


def main():
    """Fonction principale."""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <config.json>                    # Valider un fichier")
        print(f"  {sys.argv[0]} <config1.json> <config2.json>   # Comparer deux fichiers")
        sys.exit(1)
    
    config1_path = sys.argv[1]
    
    if len(sys.argv) == 2:
        # Validation simple
        success = load_and_validate_config(config1_path)
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 3:
        # Comparaison
        config2_path = sys.argv[2]
        
        # Valider les deux fichiers d'abord
        print("🔍 VALIDATION DU PREMIER FICHIER:")
        success1 = load_and_validate_config(config1_path)
        
        print("\n🔍 VALIDATION DU SECOND FICHIER:")
        success2 = load_and_validate_config(config2_path)
        
        if success1 and success2:
            print("\n")
            compare_configs(config1_path, config2_path)
        else:
            print("\n❌ Impossible de comparer: erreurs de validation")
            sys.exit(1)
    
    else:
        print("❌ Trop d'arguments")
        sys.exit(1)


if __name__ == "__main__":
    main()