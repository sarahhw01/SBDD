from pymol import cmd

import chempy
import numpy as np


""" Some words about terminology and PyMOL data structures

    In the documentation and help texts I use the terms 'model' and 'PyMOL object'.
    A 'model' is the internal data structure of PyMOL to represent chemical entities.
    It is provided by PyMOL's internal chemistry module Chempy.
    If used in a script, a 'model' is not automatically added to and visualized in the PyMOL GUI.

    With 'PyMOL object' I refer to 'models' that are indeed loaded into the PyMOL GUI.
    That is, you can find their entry in the list of models at the right-hand side.
    In fact, a 'PyMOL object' is a 'model' itself.

    You can load 'PyMOL objects' into a 'model'
    and you can create a new 'PyMOL object' from a model,
    that is you load the 'model' into the GUI.
"""


""" Global variable to store the Chempy model to be manipulated """
model = None


def createAtom(index, name, symbol, resn, chain, resi, pos, is_het):
    """ Create a new Chempy atom

    Creates a new Chempy atom that can be used in PyMOL models.

    Args:
        index:  the internal index of the new atom
        name:   the spelled out name of the atom's element
        symbol: the symbol of the the atom's element
        resn:   the name of the residue to which the new atom is added (3-letter-code)
        chain:  the chain to which the new atom is added
        resi:   the index of the atom's residue
        pos:    the position (i.e. x,y,z coordinates) of the atom
        is_het: indicator if it is an hetero-atom

    Returns:
        atom:   the newly created atom

    """

    atom = chempy.Atom()
    atom.index = index
    atom.name = name
    atom.symbol = symbol
    atom.resn = resn
    atom.chain = chain
    atom.resi = resi
    atom.resi_number = int(resi)
    atom.coord = pos
    atom.hetatm = is_het

    return atom


def createBond(index1, index2, order=1):
    """ Create a new Chempy bond

    Creates a new Chempy bond that can be used in PyMOL models.

    Args:
        index1: index of the first incidental atom
        index2: index of the second incidental atom
        order:  bond order

    Returns:
        bond:   the newly created bond

    """

    bond = chempy.Bond()
    bond.order = order
    bond.index = [index1, index2]

    return bond


def addHydrogen(atom, pos):
    """ Attach a new hydrogen to the given atom

    Creates a new hydrogen atom with Cartesian coordinates given by pos.
    Adds the new hydrogen atom to the global model.
    Connects the new hydrogen atom to the given atom.

    Args:
        atom: atom to which the new hydrogen shall be attached (bonded)
        pos:  List of cartesian coordinates of the new hydrogen atom

    Returns:

    """

    global model

    if model is None:
        print("Error: no global model available for hydrogen attachment.")
        sys.exit(0)

    hydrogen = createAtom(model.nAtom - 1, "Hydrogen", "H", str(atom.resn) , "", str(atom.resi), (pos[0], pos[1], pos[2]), False)
    model.add_atom(hydrogen)

    bond = createBond(atom.index - 1, model.nAtom - 1)
    model.add_bond(bond)

    model.update_index()


def addHydrogensTrigonalPlanar(atom, neighbors):
    """ Add hydrogen to given atom with trigonal planar geometry

    Calculate the position of one hydrogen to be attached to the given atom.
    The given atom, the neighboring atoms and the new hydrogen shall have a trigonal planar geometry
    (an example for such a geometry is the backbone nitrogen and its neighbors).
    The new hydrogen shall finally generated and attached using the function 'addHydrogen(atom, pos)'.
    The latter function is already implemented. No work here!

    Args:
        atom:       atom to which the new hydrogen shall be attached (i.e. bonded)
        neighbors:  numpy array with Cartesian coordinates of all neighbors of the given atom

    Returns:

    """
    # check if atom has 2 neighbors
    if len(neighbors) != 2:
        raise ValueError(f"Got {len(neighbors)} neighbors, need 2 for trigonal planar!")
    
    # get array of atom's coordinates
    center_atom = np.array(atom.coord)

    # create vectors from center atom to each neighbor
    v1 = neighbors[0] - center_atom
    v2 = neighbors[1] - center_atom

    # normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # compute average of the normalized vectors and use "negative direction"
    v3 = -(v1_norm + v2_norm)
    v3_norm = v3 / np.linalg.norm(v3)

    # determine position of hydrogen to be added
    pos_h = center_atom + v3_norm * 1.0 # bond length N-H = 1.0 Å
    addHydrogen(atom, pos_h)

    # Help
    # - You can solve this problem using standard linear algebra.
    # - Numpy is already imported ;)
    # - It is very helpful to look up 'trigonal planar' and to visually inspect a protonated backbone!
    # - Please assume a bond-length of N-H = 1.0


def addHydrogensTetrahedral(atom, neighbors):
    """ Add hydrogen to given atom with tetrahedral geometry

    Calculate the position of one hydrogen to be attached to the given atom.
    The given atom, the neighboring atoms and the new hydrogen shall have a tetrahedral geometry
    (an example for such a geometry is the (sp3-hybridized) C-alpha atom and its neighbors).
    The new hydrogen is finally generated and attached using the function 'addHydrogen(atom, pos)'.

    Args:
        atom:       atom to which the new hydrogen shall be attached (i.e. bonded)
        neighbors:  numpy array with Cartesian coordinates of all neighbors of the given atom

    Returns:

    """

    # Help
    # - You can solve this problem using standard linear algebra.
    # - Numpy is already imported ;)
    # - It is very helpful to look up 'tetrahedral' and to visually inspect a protonated backbone!
    # - Please assume a bond-length of C-H = 1.1

    # Task 1
    # - Please prepare three cases to treat the given atom depending on the number of current neighbors.
    #   We will use this function in the next assignment sheet to treat further tetrahedral carbons.

    # get array of atom's coordinates
    center_atom = np.array(atom.coord)
    num_neighbors = len(neighbors)

    if num_neighbors == 4:
        # no hydrogen to add
        pass

    if num_neighbors > 4:
        raise ValueError("Already > 4 neighbors present. Tetrahedral not defined!")
    
    # create vector from center atom to first neighbor atom
    v1 = neighbors[0] - center_atom
    # normalize the vector
    v1_norm = v1 / np.linalg.norm(v1)

    if num_neighbors == 1:
        # pass for now
        pass

    if num_neighbors > 1:
        v2 = neighbors[1] - center_atom
        v2_norm = v2 / np.linalg.norm(v2)

    if num_neighbors == 2:
        # pass for now (case for glycine)
        pass

    if num_neighbors > 2:
        v3 = neighbors[2] - center_atom
        v3_norm = v3 / np.linalg.norm(v3)

    # case that applies to all C-alpha backbone atoms except glycine 
    if num_neighbors == 3:
        # add 1 hydrogen to complete tetrahedron
        # compute average of the normalized vectors and use "negative direction"
        v4 = -(v1_norm + v2_norm + v3_norm)
        v4_norm = v4 / np.linalg.norm(v4)

        # determine position of hydrogen to be added
        pos_h = center_atom + v4_norm * 1.1 # bond length C-H = 1.1 Å
        addHydrogen(atom, pos_h)


@cmd.extend
def createPeptide(pep='AFKGH'):
    """ Generates a peptide in PyMOL

    Generates a peptide corresponding to the given sequence as an object named 'pep'.
    Extracts the hydrogens from 'pep' into a new object named 'hyd'.
    Shows the object 'pep' in stick-representation only.

    Args:
        pep: primary sequence in one-letter code (default shall be 'AFKGH')

    Returns:
        Tuple of names ('pep', 'hyd') for reference

    """
    # build peptide entity from sequence using fab
    cmd.fab(pep, name='pep')

    # remove previous hyd object if present
    cmd.delete('hyd')

    # extract hydrogens from 'pep' into 'hyd'
    # we don't use cmd.extract since this removes the hydrogens from pep
    # instead, we select hydrogens and store them in a new object 'hyd'
    cmd.select('hyd_sel', 'pep and hydro')
    cmd.create('hyd', 'hyd_sel')
    cmd.delete('hyd_sel')

    # show 'pep' in stick representation only
    cmd.hide('everything', 'pep')
    cmd.show('stick', 'pep')

    return 'pep', 'hyd'

    # Helpful resources
    # - https://www.pymolwiki.org/index.php/Fab
    # - https://pymolwiki.org/index.php/Select
    # - https://pymolwiki.org/index.php/Selection_Algebra
    # - https://pymolwiki.org/index.php/Extract
    # - https://pymolwiki.org/index.php/Delete
    # - https://pymolwiki.org/index.php/Show



@cmd.extend
def sbddProtonator(object):
    """ Protonate a PyMOL protein object

    Creates a new model for which the hydrogen atoms are reconstructed.
    The protonated model is loaded into a new PyMOL object with the name '<object-name>-protonated'

    Args:
        object: name of the PyMOL object to be protonated

    Returns:

    """

    # Task 1
    # - Make the global variable 'model' accessible
    global model 
    # - Load the given PyMOL object into the global model variable (which allows independent manipulation)
    model = cmd.get_model(object)
    
    # Hints
    # - https://pymolwiki.org/index.php/Get_Model

    # Task 3
    # Loop over all atoms in the global model
    for atom in model.atom:
        # For every atom: retrieve numpy array of Cartesian coordinates of all neighboring (directly bonded) atoms
        neighbors = []
        for b in model.bond:
            if atom.index in b.index:
                # Find the index of the neighboring atom
                neighbor_idx = b.index[1] if b.index[0] == atom.index else b.index[0]
                neighbor_atom = model.atom[neighbor_idx]
                neighbors.append(np.array(neighbor_atom.coord))

        # Create an if-case to treat all backbone nitrogens (skip proline!)
        if atom.name == "N" and atom.resn != "PRO":
            # Create an if-case that allows to skip the n-terminal nitrogen (we do not treat it now)
            if atom.resi == "1":
                continue
            # For all other nitrogens call the function addHydrogensTrigonalPlanar(...)
            else:
                addHydrogensTrigonalPlanar(atom, neighbors)

        # Create an elif-case to treat all C-alpha atoms
        elif atom.name == "CA":
            # For all C-alpha atoms call the function addHydrogensTetrahedral(...)
            addHydrogensTetrahedral(atom, neighbors)

        # Create an else-case to treat all other atoms
        else:
            print(f"Untreated PDB atom: {atom.name} in {atom.resn} {atom.resi}")
    
    # Hints
    # - Before diving into addHydrogensTrigonalPlanar(...) and addHydrogensTetrahedral(...) please finish task 3.
    #   That is, make sure your loop, coordinate retrieval, and if-cases work properly.
    #
    # - https://pymolwiki.org/index.php/Get_Model
    # - https://pymolwiki.org/index.php/Get_coords
    # - https://pymolwiki.org/index.php/Selection_Algebra

    # Task 2
    # - Make sure the name '<object-name>-protonated' for the new PyMOL object does not exist. Remove it if necessary.
    name = f"{object}-protonated"
    cmd.delete(name)

    # - Load the protontated global model into a new PyMOL object with the name '<object-name>-protonated'.
    cmd.load_model(model, name)

    # - Show only the stick-representation for this new PyMOL object.
    cmd.hide("everything", name)
    cmd.show("stick", name)
    
    # Hints
    # - https://pymolwiki.org/index.php/Delete
    # - https://pymolwiki.org/index.php/Load_Model
