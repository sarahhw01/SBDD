import sys
from molecule import Molecule, Point3D, Atom

__author__ = 'Luis de la Garza'


def error(*objs):
    print(*objs, file=sys.stderr)

def print_loaded_string(molecules, file_name):
    loaded_string = "Loaded %d molecule" % len(molecules)
    if (len(molecules) > 1 or len(molecules) == 0): loaded_string += "s"
    loaded_string += " from %s." % file_name
    print(loaded_string)

class SDFReader():
    def __init__(self):
        pass

    @staticmethod
    def is_header(line_number):
        return line_number >= 1 and line_number <= 3

    @staticmethod
    def is_counts_line(line_number):
        return line_number == 4

    @staticmethod
    def is_atom_table(line_number, n_atoms):
        return line_number <= (4 + n_atoms)

    @staticmethod
    def is_bond_table(line_number, n_atoms, n_bonds):
        return line_number <= (4 + n_atoms + n_bonds)

    @staticmethod
    def is_properties_block(line_number, n_atoms, n_bonds):
        return line_number > (4 + n_atoms + n_bonds)

    @staticmethod
    def load(file_name):
        # we read a list of molecules
        read_molecules = []
        sdf_file = None
        try:
            sdf_file = open(file_name, "r")
        except IOError as e:
            # file not found or something... not much we could do other than print an error and return an empty
            # molecule list
            error(">>> Error while opening %s" % file_name)
            error(">>> I/O error code %d: %s" % (e.errno, e.strerror))
            error(">>> No molecules were loaded.")
            return read_molecules

        line_number = 1
        molecule_name = ""
        n_atoms = 0
        n_bonds = 0
        molecule = ''
        absolute_line = 1
        property_name = None
        reading_property = False
        for line in sdf_file:
            # decide where are we based on the line number
            if SDFReader.is_header(line_number):
                # we append the header lines to the name of the molecule
                if line_number > 1:
                    molecule_name += ','
                molecule_name += line.strip()
            elif SDFReader.is_counts_line(line_number):
                # we can create the molecule and add its name
                molecule = Molecule(molecule_name)
                # reset the name
                molecule_name = ''
                # extract the number of atoms and bonds
                # number of atoms are the first three characters and number of bonds the next three
                n_atoms = int(line[0:3])
                n_bonds = int(line[3:6])
            elif SDFReader.is_atom_table(line_number, n_atoms):
                # the coordinates are given in the first 30 characters, 10 per each dimension
                x_pos = float(line[0:10])
                y_pos = float(line[10:20])
                z_pos = float(line[20:30])
                # then the name of the atom
                symbol = line[30:33].strip()
                # insert the atom in the molecule
                molecule.add_atom(Atom(symbol, Point3D(x_pos, y_pos, z_pos), 1.0))
            elif SDFReader.is_bond_table(line_number, n_atoms, n_bonds):
                # the first three characters represent the first atom
                atom_one_index = int(line[0:3])
                # the 4th-6th characters represent the second atom
                atom_two_index = int(line[3:6])
                # the 7th-9th characters represent the bond type
                bond_type = int(line[6:9])
                # remember that in python indices start from 0, while in mol files, they start from 1!!!
                molecule.add_bond(atom_one_index - 1, atom_two_index - 1, bond_type)
            elif SDFReader.is_properties_block(line_number, n_atoms, n_bonds):
                if line == "M  END":
                    # ignore
                    pass
                elif line.startswith('>'):
                    # property
                    reading_property = True
                    # this is what the line looks like_
                    # > <PUBCHEM_COMPOUND_CID>
                    property_name = line[3:line.rfind('>')]
                elif reading_property:
                    # the name has been read, we take the whole line as the value (remove line break)
                    # right now, no support for multi-line property values is available...
                    molecule.add_property(property_name, line.rstrip('\n'))
                    reading_property = False
                elif line.strip().startswith("$$$$"):
                    # end of molecule!
                    read_molecules.append(molecule)
                    line_number = 0

            # done reading this line
            line_number += 1
            absolute_line += 1

        # done reading file, close it
        try:
            sdf_file.close()
        except IOError:
            # ignore the error
            pass
        print_loaded_string(read_molecules, file_name)
        return read_molecules

    if __name__ == "__main__":
        from sdf_reader import SDFReader
        if len(sys.argv) < 2:
            error("Not enough parameters provided.")
            error("Usage:")
            error("  python sdf_reader.py <input-sdf>")
            sys.exit(-1)
        file_name = sys.argv[1]
        molecules = SDFReader.load(file_name)
        print("Read %d molecules from %s" % (len(molecules), file_name))
        for molecule in molecules:
            print("  Molecule %s has %d atoms" % (molecule.name, len(molecule)))
            for atom in molecule:
                print("    found atom %s with position (%f, %f, %f) and mass %f" % (atom.symbol, atom.position.x, atom.position.y, atom.position.z, atom.atomic_mass))
                for bond in molecule.get_bonds_for_atom(atom):
                    print("      bond %s-%s of type %i " % (bond.atom_one.symbol, bond.atom_two.symbol, bond.bond_type))
