#!/usr/bin/env python3
"""
Example script showing how to test the pipeline with your VASP files.

This script demonstrates the complete workflow for testing our DFT→GNN→QNN
pipeline using your VASP files and formation energies.
"""

import sys
import os
sys.path.append('src')

def main():
    print("🔬 VASP Pipeline Testing Workflow")
    print("=" * 50)
    
    print("\n📋 STEP 1: Prepare your VASP files")
    print("   - Place all your VASP files (POSCAR, CONTCAR, *.vasp) in a directory")
    print("   - Example: /path/to/your/vasp/files/")
    print("   - Files should contain the final relaxed structures from VASP")
    
    print("\n📋 STEP 2: Prepare formation energies")
    print("   - Create a file with structure_id and formation_energy pairs")
    print("   - Supported formats: .txt, .csv, .json")
    print("   - Example formats:")
    print("     structure_001  -2.45")
    print("     structure_002  -1.23")
    print("     structure_003  -3.67")
    
    print("\n📋 STEP 3: Create sample energy file (optional)")
    print("   python3 scripts/test_vasp_pipeline.py create-sample-energy-file \\")
    print("     --vasp-dir /path/to/your/vasp/files \\")
    print("     --output my_energies.txt")
    print("   # Then edit my_energies.txt with your actual formation energies")
    
    print("\n📋 STEP 4: Test the pipeline")
    print("   python3 scripts/test_vasp_pipeline.py test-pipeline \\")
    print("     --vasp-dir /path/to/your/vasp/files \\")
    print("     --energy-file my_energies.txt \\")
    print("     --output vasp_test_results.json \\")
    print("     --use-delta-head")
    
    print("\n📋 STEP 5: Analyze results")
    print("   - Check the output JSON file for detailed results")
    print("   - Look at MAE, RMSE, and relative errors")
    print("   - Compare with chemical accuracy (< 0.043 eV)")
    
    print("\n🎯 Expected Results:")
    print("   - MAE < 0.1 eV: Excellent accuracy")
    print("   - MAE < 0.5 eV: Good accuracy") 
    print("   - MAE < 1.0 eV: Moderate accuracy")
    print("   - MAE > 1.0 eV: Needs improvement")
    
    print("\n🔧 Troubleshooting:")
    print("   - If errors occur, check VASP file format")
    print("   - Ensure formation energies are in eV")
    print("   - Verify structure IDs match between files")
    print("   - Check that ASE can read your VASP files")
    
    print("\n📊 Example Output:")
    print("   🎯 VASP Pipeline Test Results:")
    print("      Structures processed: 25")
    print("      MAE: 0.234 eV")
    print("      RMSE: 0.456 eV") 
    print("      Mean relative error: 12.3%")
    print("      ✅ GOOD accuracy!")
    
    print("\n" + "=" * 50)
    print("🚀 Ready to test your VASP files!")

if __name__ == "__main__":
    main()


