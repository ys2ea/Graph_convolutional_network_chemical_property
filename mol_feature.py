# Use RDkit to convert SMILES string to a set of atom and bond features.
# Also load and preprocess data, do simple analysis of elements

import operator
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols

#list of all elements(except H), which is the result of running list_atom_types function below

#all_elements = ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Al', 'Sn', 'Hg', 'As', 'Cr', 'Zn', 'Na', 'Fe', 'H', 'Se', 'Cu', 'Ba', 'Au', 'Co', 'Ca', 'Sb', 'In', 'Ni', 'K', 'Cd', 'Ti', 'Mn', 'Pt', 'Mg', 'Zr', 'Gd', 'Li', 'Bi', 'Pd', 'Tl', 'Ag', 'Mo', 'V', 'Nd', 'Yb', 'Pb', 'Dy', 'Sr', 'Be', 'Ge']

#35 types
all_elements = ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Al', 'Sn', 'Hg', 'As', 'Cr', 'Zn', 'Na', 'Fe', 'H', 'Se', 'Cu', 'Ba', 'Au', 'Co', 'Ca', 'Sb', 'In', 'Ni', 'K', 'Cd', 'Ti', 'Mn', 'Pt', 'other']

length_atom_feature = 53
length_bond_feature = 5

#load data
def load_data(filename):
    smiles = np.genfromtxt(filename, comments=None, dtype=np.str, delimiter=",", usecols=0) 
    cols = [x for x in range(1,13)]
    labels = np.genfromtxt(filename, comments=None, delimiter=",", usecols=cols)   

    #remove first row which is comment
    smiles = smiles[1:]
    labels = labels[1:, :]
    return [smiles, labels]

    
def list_atom_types(s_data):
    ele = {}
    for s in s_data:
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ele:
                ele[atom.GetSymbol()] += 1
            else:
                ele[atom.GetSymbol()] = 1
            
    ele_sort = sorted(ele.items(), key=operator.itemgetter(1),reverse=True)
    #print(ele_sort)
    
    #also create an 'other' type, if an element appears less than 3 times it's put in this type
    elements = [ele_sort[i][0]  for i in range(len(ele_sort)) if ele_sort[i][1]>0]
    print("Len: ", len(elements))
    return elements + ['other']

# From smiles return atom feature array and bond feature array
def build_feature(mol_smiles):
    mol = Chem.MolFromSmiles(mol_smiles)
    
    atom_features = []
    bond_features = np.zeros((len(mol.GetAtoms()), length_bond_feature))
    connection = np.zeros((len(mol.GetAtoms()), len(mol.GetAtoms())))
    
    for atom in mol.GetAtoms():
        atomtype = [0 for i in range(len(all_elements))]
        if atom.GetSymbol() not in all_elements[:-1]: atomtype[-1] = 1
        else:
            for i in range(len(all_elements)):
                if atom.GetSymbol()==all_elements[i]: atomtype[i] = 1
        #print("L t: ", len(atomtype))
        atomdegree = [1 if atom.GetDegree() == i else 0 for i in range(6)]
        
        atomHnumber = [1 if atom.GetTotalNumHs() == i else 0 for i in range(5)]
        
        atomvalence = [1 if atom.GetImplicitValence() == i else 0 for i in range(6)]
        
        atomaromatic = [atom.GetImplicitValence()]
        
        atom_features.append(atomtype + atomdegree + atomHnumber + atomvalence + atomaromatic)
        
    for bond in mol.GetBonds():
        bid = bond.GetBeginAtom().GetIdx()
        eid = bond.GetEndAtom().GetIdx()
        
        bt = bond.GetBondType()
        bf = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, \
                     bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
        
        bond_features[bid] += bf
        connection[bid, eid] = 1
        connection[eid, bid] = 1  
    return [atom_features, bond_features, connection]

# Build a mini batch, using padding to make all molecules the same size

def build_batch_feature(list_smiles):
    batch_atom = []
    batch_bond = []
    batch_connect = []
    max_length = 0
    #find the largest molecule length
    #then using padding to make all molecules the same length
    for smiles in list_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() > max_length:
            max_length = mol.GetNumAtoms()
        
    for smiles in list_smiles:
        l = Chem.MolFromSmiles(smiles).GetNumAtoms()
        afb = np.zeros((max_length, length_atom_feature))
        bfb = np.zeros((max_length, length_bond_feature))
        cfb = np.zeros((max_length, max_length))
        
        [af, bf, cf] = build_feature(smiles)
        
        afb[0:l,0:length_atom_feature] = af
        bfb[0:l,0:length_bond_feature] = bf
        cfb[0:l,0:l] = cf
        
        batch_atom.append(afb)
        batch_bond.append(bfb)
        batch_connect.append(cfb)
        
    return [np.array(batch_atom), np.array(batch_bond), np.array(batch_connect)]
    
    
def circular_fps(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fpstr = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024).ToBitString()
        fps.append(list(fpstr))
    return np.array(fps, dtype=int)
    
[smiles, l] = load_data('data.csv')   

