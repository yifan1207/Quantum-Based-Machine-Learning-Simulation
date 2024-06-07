# Import numpy for operation   
import numpy as np  
 
# Define a function to convert PDB files to numerical arrays
def pdb_to_array(pdb_file):
  # Read the pdb file and extract the coordinates of the atoms
  with open(pdb_file, "r") as f:
    lines = f.readlines()
    coords = []
    for line in lines:
      if line.startswith("ATOM"):
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append([x, y, z])
  # Convert the list of coordinates to a numpy array
  array = np.array(coords)
  # Return the array
  return array

# Define a function to compare two arrays and check if they are equal
def compare_arrays(array1, array2):
  # Check if the arrays have the same shape
  if array1.shape != array2.shape:
    return False
  # Check if the arrays have the same elements
  else:
    return np.allclose(array1, array2)

# Define a function to compare two lists of PDB files and find the repeated ones
def compare_pdbs(pdb_list1, pdb_list2):
  # Initialize an empty list to store the repeated PDB files
  repeated_pdbs = []
  # Loop through the first list of PDB files
  for pdb1 in pdb_list1:
    # Convert the PDB file to an array
    array1 = pdb_to_array(pdb1)
    # Loop through the second list of PDB files
    for pdb2 in pdb_list2:
      # Convert the PDB file to an array
      array2 = pdb_to_array(pdb2)
      # Compare the two arrays and check if they are equal
      if compare_arrays(array1, array2):
        # If they are equal, add the PDB file to the repeated list
        repeated_pdbs.append(pdb1)
        # Break the inner loop
        break
  # Return the list of repeated PDB files
  return repeated_pdbs

# Define two lists of PDB files generated by quantum simulation and quantum ml
qs_pdbs = ["qs_pdb1.pdb", "qs_pdb2.pdb", "qs_pdb3.pdb", "qs_pdb4.pdb"]
qml_pdbs = ["qml_pdb1.pdb", "qml_pdb2.pdb", "qml_pdb3.pdb", "qml_pdb4.pdb"]

# Call the compare_pdbs function and print the result
repeated_pdbs = compare_pdbs(qs_pdbs, qml_pdbs)
print("The repeated PDB files are:", repeated_pdbs)
#for int...
