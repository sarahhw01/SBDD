'''
-------------------------------------------------------------------------------
 Name:        sbdd-hg-template.py
 Purpose:     SBDD assignment 5

 Author:      Philipp Thiel

 Created:     06/05/2023
 Copyright:   (c) Philipp Thiel 2023
 Licence:     BSD3
-------------------------------------------------------------------------------
'''

from __future__ import print_function

from pymol import cmd

import time
import math
from scipy.spatial import distance



class HashGrid:
    """ This class implements a HashGrid for efficient neighbour searching
    of atoms stored in that data structure.
    """
    
    def __init__(self, prot):
        """ The __init__ method is called when a HashGrid instance is created.
        """

        # Spacing of the grid cells in Angstrom
        self.spacing = 1.0

        # The smalles coordinate value for every axis (x, y, z)
        self.minCoords = [0.0, 0.0, 0.0]

        # The largest coordinate value for every axis (x, y, z)
        self.maxCoords = [0.0, 0.0, 0.0]

        # The dimension of every axis, i.e. length it (x, y, z)
        self.dimensions = [0, 0, 0]

        # The HashGrid, which is not yet initialized
        self.hashgrid = None

        # Setup the HashGrid class
        self.setupHashGrid(prot)

        # Insert all atoms into the HashGrid
        self.insertAtoms(prot)


    def setupHashGrid(self, prot): # Step 2, 4pt
        """ Setup the HashGrid class.

        The method needs to determine and store
        - the smallest coordinate value for every axis (minCoords)
        - the largest coordinate value for every axis (maxCoords)
        - the required dimensions of the HashGrid (dimensions)

        Finally, the HashGrid must be initialized. The HashGrid needs
        to be a 3D array whose cells can be accessed using integer indices
        and which can store the indices of PyMOL atoms.

        Parameters
        ----------
        prot : PyMOL model object
            The protein object from the PyMOL GUI
        """
        # Extract all coordinates from the protein atoms
        coords = [atom.coord for atom in prot.atom]

        # Calculate min and max coordinates for every axis x, y, z
        self.minCoords = [min(c[i] for c in coords) for i in range(3)]
        self.maxCoords = [max(c[i] for c in coords) for i in range(3)]

        # Calculate dimensions (grid cells) based on defined spacing
        self.dimensions = [int(math.ceil((self.maxCoords[i] - self.minCoords[i]) / self.spacing)) + 1 for i in range(3)]
        
        # initialize the HashGrid based on the dimensions of the given model object
        self.hashgrid = [[[[] for _ in range(self.dimensions[2])]
                         for _ in range(self.dimensions[1])]
                         for _ in range(self.dimensions[0])]


    def coordinatesToIndex(self, coordinates): # Step 3, 4pt
        """ Calculate the HashGrid index for given coordinates.

        For a single triplet of atom coordinates (real valued) this method should
        calculate the correspoding index into the HashGrid where the atom is stored.
        This is basically a mapping function which is required because your HashGrid
        has integer indices and the origin of it has index (0, 0, 0).

        Parameters
        ----------
        coordinates : Triplet of coordinates for a single atom
            The (x, y, z) coordinates for which the HashGrid index should be calculated

        Returns
        -------
        list
            Triplet (i, j, k) of HashGrid indices (the 'index')
        """
        # Idea: grid origin (0,0,0) corresponds to the min coords of an atom
        # Shift the input coordinate by substracting the minCoords to get a relative position
        # Divide by spacing to know, which grid cell contains the coordinate
        # Use floor to convert float numbers to integers

        # For each axis substract the minimum coordinate (grid origin), and divide by the spacing and floor to receive the integer index
        indices = [int(math.floor((coordinates[i] - self.minCoords[i]) / self.spacing)) for i in range(3)]
        return indices


    def insertAtoms(self, prot): # Step 4, 2pt
        """ Insert all atoms of 'prot' into the HashGrid.

        The method iterates over all atoms of 'prot', calculates their HashGrid indices,
        and inserts their PyMOL atom indices together with their coordinates (tuple!)
        into their corresponding HashGrid cells.

        Parameters
        ----------
        prot : PyMOL model object
            The protein object from the PyMOL GUI
        """

        for atom in prot.atom:
            atom_coords = atom.coord
            atom_coords = (atom_coords[0], atom_coords[1], atom_coords[2])
            atom_indices = self.coordinatesToIndex(atom_coords)

            # get the hash grid indices
            i = atom_indices[0]
            j = atom_indices[1]
            k = atom_indices[2]

            # Insert the atom index into the corresponding position in the HashGrid
            self.hashgrid[i][j][k].append((atom.index, atom_coords))


    def getNeigbourAtoms(self, query, radius): # Step 5, 3pt
        """ Retrieve all atoms within given radius around the given query coorindates.

        For a given query, which is a triplet of (x, y, z) coordinates the method
        needs search the relevant HashGrid cells in order to identify all atoms
        that are canidates to lie within the given radius around the query.

        Parameters
        ----------
        query: Triplet of query coordinates
            The (x, y, z) coordinates of the query
        radius : int
            The radius to search for neighboring atoms

        Returns
        -------
        list
            List of PyMOL atom indices of the identified candidate atoms
        """
        # Turn the query coordinates x,y,z into grid indices, to find them in the HashGrid
        query_coords = self.coordinatesToIndex(query)
        q_i = query_coords[0]
        q_j = query_coords[1]
        q_k = query_coords[2]

        # Calculate the "search space" for this atom: how many cells around the query cell do we search in each direction
        # To calculate this we use the radius and the spacing value
        search_range = int(math.ceil(radius/self.spacing))

        # Define the neighbours as a set
        neighbours = set()

        # Iterate over the neighbouring cells around our query coords
        # start along i direction
        for i in range(q_i-search_range, q_i+search_range+1):
            # if indices are out of bounds, skip them
            if (i < 0) or (i >= self.dimensions[0]):
                continue
            
            # along j direction
            for j in range(q_j-search_range, q_j+search_range+1):
                # if indices are out of bounds, skip them
                if (j < 0) or (j >= self.dimensions[0]):
                    continue

                # along k direction
                for k in range(q_k-search_range, q_k+search_range+1):
                    # if indices are out of bounds, skip them
                    if (k < 0) or (k >= self.dimensions[0]):
                        continue

                    # Go over all atoms in the current cell that is within our search range
                    for atom_index, atom_coords in self.hashgrid[i][j][k]:
                        # compute the euclidean distance between the query atom and the atom corresponding to the current cell
                        euc_dist = distance.euclidean(query, atom_coords)
                        if euc_dist <= radius:
                            neighbours.add(atom_index)

        return list(neighbours)



def sbddGridNeighbourSearch(hashgrid, lig, radius): # Step 6, 3pt
    """ Find all atoms within the given radius around 'lig'.

    The method returns the indices of all atoms in the HashGrid,
    which have a distance less than <radius> Angstroms
    from any 'lig' atom.

    The implementation of this method shall use the HashGrid
    to efficiently identify the candidate atoms.

    Parameters
    ----------
    prot : PyMOL model object
        The protein object from the PyMOL GUI
    lig :  PyMOL model object
        The ligand object from the PyMOL GUI
    radius : int
        The radius to search for neighboring atoms

    Returns
    -------
    list
        Indices of identified atoms as a list of int
    """
    neighbours = set()

    # iterate through all atoms in a given ligand
    for atom in lig.atom:
        query_coords = atom.coord
        # convert the coords into the correct format to input into the getNeighbourAtoms function
        query_coords = (query_coords[0], query_coords[1], query_coords[2])
        # Find the neighbours for each ligand atom
        neighbour_atoms = hashgrid.getNeigbourAtoms(query_coords, radius)
        # add all found neighbours to our set
        neighbours.update(neighbour_atoms)

    return list(neighbours)



def sbddNaiveNeighbourSearch(prot, lig, radius): # Step 1, 4pt
    """ Find all atoms of 'prot' within the given radius around the atoms of 'lig'.

    The method returns the indices of all atoms in 'prot',
    which have a distance less than <radius> Angstroms
    from any atom in 'lig'.

    The implementation of this method (as the name implies)
    shall use the most straightforward (brute-force) approach to solve this task.

    Parameters
    ----------
    prot : PyMOL model object
        The protein object from the PyMOL GUI
    lig :  PyMOL model object
        The ligand object from the PyMOL GUI
    radius : int
        The radius to search for neighboring atoms

    Returns
    -------
    list
        Indices of identified atoms as a list of int
    """
    # initialize the list for the identified atom indices, use set to avoid duplicates
    identified_atoms = set()

    # for each atom in the ligand, and for each atom in the protein, compare the distances of atom pairs
    # select the ones with a distance smaller than the given radius
    for lig_atom in lig.atom:
        lig_coordinates = lig_atom.coord # this should give us the cartesian coordinates of an atom
        
        for prot_atom in prot.atom:
            prot_coordinates = prot_atom.coord # get cartesian coordinates of a protein atom 

            # calculate eucledean distance between the current ligand and protein atoms
            euc_dist = distance.euclidean(lig_coordinates, prot_coordinates)

            # if distance smaller than the given radius, we put these into the identified indices list
            if euc_dist <= radius:
                identified_atoms.add(prot_atom.index)

    return list(identified_atoms)



@cmd.extend # This decorator qualifies a Python method to be callable from PyMOL
def selectNeighbourAtoms(prot_name, lig_name, radius=6):
    """ PyMOL command (method) to find all atoms in 'prot_name' around the atoms in 'lig_name'.

    The final PyMOL command selects the indices of all 'prot_name' atoms,
    which have a distance less than <radius> Angstroms from any 'lig_name' atom.

    The concrete implementation is outsourced
    into a separate non-PyMOL (command) method

    Parameters
    ----------
    prot_name : str
        Name of the PyMOL object that contains the protein
    lig : str
        Name of the PyMOL object that contains the ligand
    radius : int
        The radius to search for neighboring atoms (default=3)
    """

    prot = cmd.get_model(prot_name)
    lig = cmd.get_model(lig_name)
    
    # Naive neighbour searching
    print('\n=================================')
    print('Benchmark naive search')

    start = time.time()
    neighbours = sbddNaiveNeighbourSearch(prot, lig, int(radius))
    end = time.time()

    print(f'- Selected atoms: {len(neighbours)}')
    print(f'- Runtime [s]:    {round(end - start, 6)}')


    # Generation of the HashGrid for efficient searching
    print('\n=================================')
    print("Benchmark HashGrid search:")

     # Setting up the HashGrid
    start = time.time()
    hg = HashGrid(prot)
    end = time.time()
    runtime_a = end - start

    print(f'- Runtime HashGrid setup [s]: {round(runtime_a, 6)}')

    # Neighbour searching using the HashGrid
    start = time.time()
    neighbours = sbddGridNeighbourSearch(hg, lig, int(radius))
    end = time.time()
    runtime_b = end - start

    print(f'- Selected atoms:             {len(neighbours)}')
    print(f'- Runtime [s]:                {round(runtime_b, 6)}')
    print(f'- Speedup:                    {round(runtime_a / runtime_b, 3)}')
    print()

    '''
    ###################
    LEARNING OBJECTIVE: PyMOL API method 'select'
       Please inform yourself about this method in order to understand what it does.
    '''
    cmd.select("index " + ",".join( [str(i) for i in neighbours]), quiet=1)
