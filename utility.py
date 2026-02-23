# ---------------------------------
# 1️⃣ Utility Functions
# ---------------------------------

def create_experiment_folder(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config(cfg, exp_dir):
    config_path = os.path.join(exp_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(cfg).items():
            f.write(f"{key}: {value}\n")

def save_checkpoint(model, optimizer, epoch, exp_dir, is_best=False):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Save latest model
    torch.save(checkpoint, os.path.join(exp_dir, "model.pth"))

    # Save best model separately
    if is_best:
        torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))