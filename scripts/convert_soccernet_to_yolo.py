import os
import glob
import configparser
from tqdm import tqdm

def convert_to_yolo(data_root):
    # Find all sequences in train and val
    sequences = glob.glob(os.path.join(data_root, "images", "*", "SNMOT-*"))
    
    print(f"Found {len(sequences)} sequences.")
    
    for seq_dir in tqdm(sequences):
        # Read seqinfo.ini
        seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
        if not os.path.exists(seqinfo_path):
            print(f"Skipping {seq_dir}: seqinfo.ini not found")
            continue
            
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        try:
            im_width = int(config['Sequence']['imWidth'])
            im_height = int(config['Sequence']['imHeight'])
        except KeyError:
            # Fallback or try to read from gameinfo.ini or assume 1920x1080
            im_width = 1920
            im_height = 1080
            
        # Read gt.txt
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            print(f"Skipping {seq_dir}: gt.txt not found")
            continue
            
        # Prepare output directory (labels should be in 'labels' dir, mirroring 'images')
        # images/train/SNMOT-XXX/img1 -> labels/train/SNMOT-XXX/img1
        img_dir_rel = os.path.relpath(os.path.join(seq_dir, "img1"), os.path.join(data_root, "images"))
        label_dir = os.path.join(data_root, "labels", img_dir_rel)
        os.makedirs(label_dir, exist_ok=True)
        
        # Read all lines
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            
        # Group by frame
        frame_data = {}
        for line in lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            # Heuristic for class
            # Ball is usually small (approx 13x13)
            # Players are larger
            if w < 30 and h < 30:
                class_id = 0 # Ball
            else:
                class_id = 1 # Player
                
            # Normalize
            center_x = (x + w / 2) / im_width
            center_y = (y + h / 2) / im_height
            norm_w = w / im_width
            norm_h = h / im_height
            
            # Clamp to [0, 1]
            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            if frame_id not in frame_data:
                frame_data[frame_id] = []
            
            frame_data[frame_id].append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        # Write label files
        # Images are named 000001.jpg, etc.
        for frame_id, labels in frame_data.items():
            label_filename = f"{frame_id:06d}.txt"
            label_path = os.path.join(label_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(labels))

if __name__ == "__main__":
    convert_to_yolo("data/datasets/soccernet_tracking")
