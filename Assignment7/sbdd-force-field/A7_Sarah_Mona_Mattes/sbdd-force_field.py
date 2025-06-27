from sdf_reader import SDFReader
import sys
import os
import numpy as np

__author__ = "Mona Scheurenbrand, Mattes Warning, and Sarah Hüwels"

# lookup table for parameters
parameters = {
    "stretch": {
        "r": {
            "CH": 1.09,
            "CC": 1.53,
            "HC": 1.09,
        },
        "k": {
            "CH": 340,  # kcal / (mol * Å^2) ^-1
            "CC": 240,
            "HC": 340
        }
    },
    "bend": {
        "phi": {  # φ₀ in degrees
            "HCH": 107.0,
            "HCC": 110.7,
            "CCC": 109.5,
            "CCH": 110.7,
        },
        "k": {  # kcal / (mol * rad^2)
            "HCH": 33,
            "HCC": 52,
            "CCC": 30,
            "CCH": 52,
        }
    }
}


def error(*params):
    print(*params, file=sys.stderr)


def get_coordinates(atom):
    """ Get (x,y,z) coordinates"""
    pos = atom.position

    return np.array([pos.x, pos.y, pos.z])

def angle_between(v1, v2):
    """Return the angle (in radians) between vectors v1 and v2."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)


def process_bond(bond):
    """Calculate stretch energy of a bond"""
    atom_one = bond.atom_one
    atom_two = bond.atom_two

    bond_type = f"{atom_one.symbol}{atom_two.symbol}"

    k_s = parameters['stretch']['k'][bond_type]
    r_0 = parameters['stretch']['r'][bond_type]

    cords_atom_one = get_coordinates(atom_one)
    cords_atom_two = get_coordinates(atom_two)

    r_ij = np.linalg.norm(np.abs(cords_atom_one - cords_atom_two))

    # stretch energy
    energy = k_s * (r_ij -r_0)**2

    f.write(f"    {bond_type}: {r_ij:.6f} A ({energy:.6f} kcal/mol)\n")

    return energy

def process_angle(bond_one, bond_two):
    """Calculate bend energy between two bonds (3 Atoms)"""

    b1_a1_c = get_coordinates(bond_one.atom_one)
    b1_a2_c = get_coordinates(bond_one.atom_two)
    b2_a1_c = get_coordinates(bond_two.atom_one)
    b2_a2_c = get_coordinates(bond_two.atom_two)

    # no common atom -> no angle (first atom is always common atom)
    if (b1_a1_c != b2_a1_c).any() and (b1_a2_c != b2_a1_c).any():

        return 0

    # first atoms match
    if (b1_a1_c == b2_a1_c).all():
        angle_name = f"{bond_one.atom_two.symbol}{bond_one.atom_one.symbol}{bond_two.atom_two.symbol}"

        i = b1_a2_c
        j = b1_a1_c
        k = b2_a2_c

    # second atom of bound one and first atom of bound two match
    if (b1_a2_c == b2_a1_c).all():
        angle_name = f"{bond_one.atom_one.symbol}{bond_one.atom_two.symbol}{bond_two.atom_two.symbol}"

        i = b1_a1_c
        j = b1_a2_c
        k = b2_a2_c

    k_b = parameters['bend']['k'][angle_name]
    phi_0 = parameters['bend']['phi'][angle_name]

    # convert from degrees to radians
    phi_0 = np.deg2rad(phi_0)

    # bond vectors between atoms
    v_ij = i - j
    v_kj = k -j

    # get angle between bond vectors
    phi_ij = angle_between(v_ij, v_kj)

    # calculate bend energy
    energy = k_b * (phi_ij - phi_0)**2

    # write to file
    f.write(f"    {angle_name}: {phi_ij:.6f} radians ({energy:.6f} kcal/mol)\n")

    return energy


if __name__ == "__main__":
    # check if we have everything we need
    if len(sys.argv) < 2:
        error("Not enough arguments provided.")
        error("Usage:")
        error("  python force_field.py <input-sdf>")
        sys.exit(-1)
    # the first given argument is the sdf file that we will parse and use
    file_name = sys.argv[1]
    molecules = SDFReader.load(file_name)

    # get clean name of input file
    name = file_name.split("/")[-1].split(".")[0]
    output_file = f"{name}.out"

    # open file and write header
    with open(output_file, "w") as f:
        f.write(f"Loaded {len(molecules)} molecule from {file_name}.\n")
        f.write("======================================\n")

        # Process molecules
        for mol in molecules:
            f.write(f"  == Processing molecule: {mol.name}\n")
            f.write("  Bonds\n")

            stretch_energy = 0
            bend_energy = 0
            angle_count = 0

            for bond in mol.bonds:
                stretch_energy += process_bond(bond)

            f.write("  Angles\n")

            # pairwise comparison
            for i in range(len(mol.bonds) - 1):
                for j in range(1, len(mol.bonds) - i):

                    energy = process_angle(mol.bonds[i], mol.bonds[i+j])

                    # we return 0 if no common atom is present -> no count
                    if energy != 0:
                        angle_count += 1

                    bend_energy += energy


            f.write(f"  == Bond count: {len(mol.bonds)}\n")
            f.write(f"  == Angle count: {angle_count}\n")

            f.write(f"  == stretch energy: {stretch_energy:.6f} kcal/mol\n")
            f.write(f"  == bend energy:    {bend_energy:.6f} kcal/mol\n")
            f.write(f"  == total energy:   {stretch_energy + bend_energy:.6f} kcal/mol\n")
            f.write("======================================\n")


    """
    TODO

    Your implementation goes here!

    TODO
    """
