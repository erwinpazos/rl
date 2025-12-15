import os
print("Current dir:", os.getcwd())
print("Parent dir exists:", os.path.exists("../basic_control"))
print("Model exists:", os.path.exists("../basic_control/behavioral_cloning_model.pth"))
print("Absolute path:", os.path.abspath("../basic_control/behavioral_cloning_model.pth"))