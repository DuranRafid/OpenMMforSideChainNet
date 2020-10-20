from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from pdbfixer import PDBFixer
import sidechainnet as scn
import pickle
import io

class OpenMMSCN(object):
    """ Performs MD Simulations to SideChainNet Data"""

    def __init__(self,path_to_pickle='sidechainnet_casp12_30.pkl'):
        self.data = self._load_data(path_to_pickle)

    def _load_data(self, path_to_pickle):
        with open(path_to_pickle, "rb") as f:
            data = pickle.load(f)
        return data

    def get_PDBs(self):
        return list(self.data.keys())

    def _get_pdb_object(self,PDBID):
        """
             :param PDBID: The PDBID which is a key of self.data dictionary
             :return: Returs the PDBFixer Object
        """
        true_coords = self.data[PDBID]['crd']
        #angles = self.data[PDBID]['ang']
        sequence = self.data[PDBID]['seq']
        sb = scn.StructureBuilder(sequence, coords=true_coords)
        pdb = PDBFixer(pdbfile=io.StringIO(sb.to_pdb_string()))
        pdb.findMissingResidues()
        pdb.findMissingAtoms()
        pdb.addMissingAtoms()
        pdb.addMissingHydrogens(7.0)
        return pdb

    def set_up_MD_env(self, PDBID):
        """
            :param PDBID: The PDBID which is a key of self.data dictionary
            :return: Sets up Molecular Dynamics Environment for given PDBID. Returns Nothing.
        """
        self.pdb = self._get_pdb_object(PDBID)
        self.forcefield = ForceField('amber14-all.xml', 'amber14/protein.ff14SB.xml')
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedMethod=NoCutoff)
        self.integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
        self.simulation = Simulation(self.modeller.topology, self.system, self.integrator)
        self.simulation.context.setPositions(self.modeller.positions)

    def get_force_per_atom(self,PDBID):
        """
             :param PDBID: The PDBID which is a key of self.data dictionary
             :return: Returns Gradient Per Atom for the PDB
        """
        if len(self.data[PDBID].keys())!= 5:
            print("The keys are not standard. %s Can not be processed" % PDBID)
            return
        self.set_up_MD_env(PDBID)
        state = self.simulation.context.getState(getForces=True)
        forces = state.getForces()
        forceNormSum = 0.0 * kilojoules ** 2 / mole ** 2 / nanometer ** 2
        for f in forces:
            forceNormSum += dot(f, f)
        forceNorm = sqrt(forceNormSum)

        #assert count == pdb.topology.getNumAtoms(), "Forces not available for all atoms"
        #print("forceNorm =", forceNorm)
        return forceNorm / self.pdb.topology.getNumAtoms()


if __name__ == '__main__':
    omscn = OpenMMSCN("C:\\CPCB\\David Koes\\sidechainnet_casp12_30.pkl")
    results = open("C:\\CPCB\\David Koes\\results.txt","a")
    print(len(omscn.get_PDBs()))
    for i in range(177,len(omscn.get_PDBs())):
        PDBID = omscn.get_PDBs()[i]
        results.write('Gradient per Atom for '+PDBID+': '+str(omscn.get_force_per_atom(PDBID))+'\n')
    results.close()




