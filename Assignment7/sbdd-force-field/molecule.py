__author__ = 'Luis de la Garza'


# class that represents a point in a three dimensional space
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __str__(self):
        return "Point3D [x=%f, y=%f, z=%f]" % (self.x, self.y, self.z)


# simple class that represents an atom
class Atom:
    # constructor
    def __init__(self, symbol, position, atomic_mass=1.0, index_in_molecule=0):
        self.symbol = symbol
        self.position = position
        self.atomic_mass = atomic_mass
        # useful when used inside a molecule
        self.index_in_molecule = index_in_molecule

    def __str__(self):
        return "Atom [symbol = %s, position = %s, mass = %f, index = %d]" % \
               (self.symbol, self.position, self.atomic_mass, self.index_in_molecule)


# bond types
class BondType:
    Single = 1
    Double = 2
    Triple = 3
    Aromatic = 4
    Single_Or_Double = 5
    Single_Or_Aromatic = 6
    Double_Or_Aromatic = 7
    Any = 8


# class that represents an atomic bond
class Bond:
    def __init__(self, atom_one, atom_two, bond_type):
        self.atom_one = atom_one
        self.atom_two = atom_two
        self.bond_type = bond_type

    def __str__(self):
        return "Bond [%s <-> %s, type %d]" % (self.atom_one, self.atom_two, self.bond_type)


class Molecule:
    def __init__(self, name):
        # there is not much we can do at init time...
        self.name = name
        # we maintain two lists, one for MoleculeAtoms and one for AtomBond
        self.atoms = []
        self.bonds = []
        # properties as a dictionary (property_name -> property_value)
        self.properties = {}

    def __getitem__(self, item):
        return self.atoms[item]

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        return self.atoms.__iter__()

    def add_atom(self, atom):
        self.atoms.append(atom)
        # change the index of the atom that was just added
        atom.index_in_molecule = len(self.atoms) - 1

    def add_bond(self, atom_one_index, atom_two_index, bond_type):
        self.bonds.append(Bond(self.atoms[atom_one_index], self.atoms[atom_two_index], bond_type))

    # return a bond by index
    def get_bond(self, index):
        return self.bonds[index]

    # return all bonds
    def get_bonds(self):
        return self.bonds

    # return the bonds in which an atom takes part
    def get_bonds_for_atom(self, atom):
        bonds = []
        for bond in self.bonds:
            if bond.atom_one == atom or bond.atom_two == atom:
                bonds.append(bond)
        return bonds

    # add a property
    def add_property(self, name, value):
        self.properties[name] = value