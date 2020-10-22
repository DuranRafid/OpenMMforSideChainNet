from openmmpdb import OpenMMPDB
import sidechainnet as scn
import pickle


class OpenMMSCN(object):
    """ Performs to SideChainNet Data"""

    def __init__(self,data,path_to_pickle=None):
        if (data is None and path_to_pickle is None) or (data is not None and path_to_pickle is not None):
            raise AssertionError("Both data and path can not be present or absent.")
        if data is None:
            self.data = self._load_data(path_to_pickle)
        else:
            self.data = data

    def _load_data(self, path_to_pickle):
        with open(path_to_pickle, "rb") as f:
            data = pickle.load(f)
        return data

    def get_PDBs(self):
        """
        Returns all the PDB IDs present in data['train']
        """
        return list(self.data['train']['ids'])

    def get_index(self,PDBID):
        """
        Returns index of PDB ID in data['train']
        """
        return self.data['train']['ids'].index(PDBID)

    def get_openmmpdb_object(self,index):
        """
             :param index: The index in data['train']
             :return: Returns an OpenMMPDB object
        """
        true_coords = self.data['train']['crd'][index]
        #angles = self.data[PDBID]['ang']
        sequence = self.data['train']['seq'][index]
        sb = scn.StructureBuilder(sequence, crd=true_coords)
        sb._initialize_coordinates_and_PdbCreator()
        pdbstr = sb.pdb_creator.get_pdb_string()
        pdb_mm = OpenMMPDB(pdbstr)
        return pdb_mm

    def get_energy_per_batch(self,start_index,batch_size=10):
        for index in range(start_index,start_index+batch_size):
            pdb_mm = self.get_openmmpdb_object(index)
            yield pdb_mm.get_potential_energy()

    def get_gradient_per_batch(self,start_index,batch_size=10):
        for index in range(start_index, start_index + batch_size):
            pdb_mm = self.get_openmmpdb_object(index)
            yield pdb_mm.get_forces_per_atoms()


if __name__ == '__main__':
    data = scn.load(casp_version=12, thinning=30)
    omscn = OpenMMSCN(data)
    for energy in omscn.get_energy_per_batch(0): print(energy)
    for gradient in omscn.get_gradient_per_batch(0): print(gradient)


