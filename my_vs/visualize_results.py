import os
import json
import glob

def check_easymocap_results(output_path):
    """Check if EasyMocap results are complete"""
    
    print("🔍 Checking EasyMocap Results Completeness")
    print("=" * 50)
    
    # Expected folders
    expected_folders = [
        'keypoints2d', 'smpl', 'vis_keypoints2d', 
        'person', 'pare'
    ]
    
    results = {}
    
    for folder in expected_folders:
        folder_path = os.path.join(output_path, folder)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, '*'))
            results[folder] = len(files)
            print(f"✅ {folder:<15}: {len(files)} files")
        else:
            results[folder] = 0
            print(f"❌ {folder:<15}: Missing")
    
    print("\n📊 Detailed Analysis:")
    print("-" * 30)
    
    # Check SMPL files specifically
    smpl_path = os.path.join(output_path, 'smpl')
    if os.path.exists(smpl_path):
        json_files = glob.glob(os.path.join(smpl_path, '*.json'))
        if json_files:
            # Check first and last file
            json_files.sort()
            first_file = json_files[0]
            last_file = json_files[-1]
            
            print(f"SMPL files: {len(json_files)}")
            print(f"First file: {os.path.basename(first_file)}")
            print(f"Last file:  {os.path.basename(last_file)}")
            
            # Check content of first file
            try:
                with open(first_file, 'r') as f:
                    data = json.load(f)
                
                print(f"\nSMPL data keys: {list(data.keys())}")
                
                if 'poses' in data:
                    poses = data['poses']
                    print(f"Poses shape: {len(poses)} persons")
                    if len(poses) > 0:
                        print(f"First person poses: {len(poses[0])} parameters")
                
                if 'shapes' in data:
                    shapes = data['shapes']
                    print(f"Shapes: {len(shapes)} persons")
                    if len(shapes) > 0:
                        print(f"First person shapes: {len(shapes[0])} parameters")
                        
                if 'Th' in data:
                    translation = data['Th']
                    print(f"Translation: {len(translation)} persons")
                    
            except Exception as e:
                print(f"Error reading SMPL file: {e}")
    
    # Check visualization files
    vis_path = os.path.join(output_path, 'vis_keypoints2d')
    if os.path.exists(vis_path):
        vis_files = glob.glob(os.path.join(vis_path, '*.jpg'))
        if vis_files:
            vis_files.sort()
            print(f"\nVisualization files: {len(vis_files)}")
            print(f"First vis: {os.path.basename(vis_files[0])}")
            print(f"Last vis:  {os.path.basename(vis_files[-1])}")
    
    print("\n🎯 Summary:")
    print("-" * 20)
    
    total_expected = 621  # Based on your range 0-620
    smpl_count = results.get('smpl', 0)
    vis_count = results.get('vis_keypoints2d', 0)
    
    if smpl_count >= total_expected:
        print("✅ SMPL processing: COMPLETE")
    else:
        print(f"⚠️  SMPL processing: {smpl_count}/{total_expected} files")
    
    if vis_count >= total_expected:
        print("✅ 2D Visualization: COMPLETE")
    else:
        print(f"⚠️  2D Visualization: {vis_count}/{total_expected} files")
    
    # Overall status
    if smpl_count >= total_expected and vis_count >= total_expected:
        print("\n🎉 RESULT: EasyMocap processing is COMPLETE!")
        print("\nNext steps:")
        print("1. Create video: ffmpeg -r 30 -i vis_keypoints2d/%06d.jpg -c:v libx264 -pix_fmt yuv420p keypoints_video.mp4")
        print("2. Use SMPL parameters for 3D applications")
        print("3. Optional: Create 3D visualization with external tools")
    else:
        print("\n⚠️  RESULT: Processing may be incomplete")
        print("Consider re-running with missing ranges or check for errors")
    
    return results

def create_summary_report(output_path):
    """Create a summary report of results"""
    results = check_easymocap_results(output_path)
    
    report_file = os.path.join(output_path, "processing_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write("EasyMocap Processing Summary\n")
        f.write("=" * 30 + "\n\n")
        
        for folder, count in results.items():
            f.write(f"{folder}: {count} files\n")
        
        f.write(f"\nProcessing completed on: {os.path.getctime(output_path)}\n")
        f.write("Files ready for further processing or visualization.\n")
    
    print(f"\n📄 Summary report saved: {report_file}")

if __name__ == "__main__":
    # Updated path to correct EasyMocap output location
    output_path = "output/sv1p"
    
    # Alternative paths to check
    possible_paths = [
        "output/sv1p",
        "EasyMocap/output/sv1p", 
        "./output/sv1p",
        "data/examples/cam0/output/sv1p"
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            print(f"✅ Found output at: {path}")
            break
    
    if found_path:
        results = check_easymocap_results(found_path)
        create_summary_report(found_path)
    else:
        print("❌ Output path not found. Tried:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nPlease check if EasyMocap has been run successfully.")
        print("Current directory:", os.getcwd())