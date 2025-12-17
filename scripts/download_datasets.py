import os
import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_soccernet():
    print("Downloading SoccerNet Tracking data...")
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/datasets/soccernet_tracking")
        # Downloading specific tracking data if available, or general data
        # Note: The 'tracking' split might need specific password or arguments depending on the version
        print("Note: SoccerNet often requires a password for full data. Checking public samples...")
        mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
    except ImportError:
        print("SoccerNet package not installed. Installing...")
        install_package("SoccerNet")
        download_soccernet()
    except Exception as e:
        print(f"Error downloading SoccerNet: {e}")

def download_teamtrack():
    print("\n[Info] TeamTrack dataset is usually hosted on Zenodo or GitHub.")
    print("Please visit: https://github.com/AtomScott/TeamTrack (or official repo)")
    print("Automatic download for TeamTrack is not standardized yet.")

def download_soccer_panoptic():
    print("\n[Info] Soccer Panoptic Segmentation dataset.")
    print("Please check official sources (often Kaggle or specialized academic servers).")

def download_soccertrack():
    print("\n[Info] SoccerTrack (GNSS) dataset.")
    print("Please check: https://github.com/SoccerTrack/SoccerTrack")

def main():
    os.makedirs("data/datasets/soccernet_tracking", exist_ok=True)
    os.makedirs("data/datasets/teamtrack", exist_ok=True)
    os.makedirs("data/datasets/soccer_panoptic", exist_ok=True)
    os.makedirs("data/datasets/soccertrack", exist_ok=True)
    
    print("Starting Dataset Download Process...")
    print("------------------------------------")
    
    # 1. SoccerNet
    download_soccernet()
    
    # 2. Others (Informational for now as they lack simple APIs)
    download_teamtrack()
    download_soccer_panoptic()
    download_soccertrack()
    
    print("\n------------------------------------")
    print("Download script finished.")
    print("NOTE: Real training requires massive GPU resources. This machine (CPU-only) is suitable for INFERENCE and lightweight fine-tuning only.")

if __name__ == "__main__":
    main()
