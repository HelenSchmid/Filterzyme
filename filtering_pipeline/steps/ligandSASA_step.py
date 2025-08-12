import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tempfile import TemporaryDirectory
from Bio.PDB import PDBParser, Select, PDBIO
import freesasa
freesasa.setVerbosity(1)
from collections import Counter
from biotite.structure.io.pdb import PDBFile
from biotite.structure import AtomArrayStack

from filtering_pipeline.steps.step import Step
from filtering_pipeline.utils.helpers import SingleLigandSelect
from filtering_pipeline.utils.helpers import get_hetatm_chain_ids, extract_chain_as_rdkit_mol, closest_ligands_by_element_composition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def select_ligand_from_smiles_via_composition(pdb_path: str | Path, substrate_smiles: str) -> str | None:
    """
    Choose the ligand whose element composition is closest to substrate_smiles,
    then return 'RESNUM:RESNAME:CHAIN' as input for fpocket. 
    """
    pdb_path = str(pdb_path)

    # find ligand chain ids
    chain_ids = get_hetatm_chain_ids(pdb_path)
    if not chain_ids:
        return None

    # get ligand mol objects
    chain_mols = []
    for ch in chain_ids:
        try:
            m = extract_chain_as_rdkit_mol(pdb_path, ch, sanitize=False)
        except Exception:
            m = None
        if m is not None:
            chain_mols.append((ch, m))

    if not chain_mols:
        return None

    # choose ligand closest by element composition
    mols_only = [m for _, m in chain_mols]
    ranked = closest_ligands_by_element_composition(mols_only, substrate_smiles, top_k=1)
    if not ranked:
        return None
    best_mol = ranked[0]

    # map back to chain id (first match)
    best_chain = next(ch for ch, m in chain_mols if m is best_mol)

    # read residue number and name for that chain 
    with open(pdb_path, "r") as f:
        pdb_file = PDBFile.read(f)
    structure = pdb_file.get_structure()
    if isinstance(structure, AtomArrayStack):
        structure = structure[0]

    mask = (structure.chain_id == best_chain)
    if not np.any(mask):
        return None

    res_ids = structure.res_id[mask] if hasattr(structure, "res_id") else structure.residue_id[mask]
    res_names = structure.res_name[mask] if hasattr(structure, "res_name") else structure.residue_name[mask]

    unique_ids = np.unique(res_ids)
    resnum = int(unique_ids.min())
    resseq  = int(np.min(res_ids))
    resname = Counter([str(r).strip() for r in res_names.tolist()]).most_common(1)[0][0]


    # return RESNUM:RESNAME:CHAIN (e.g., "1:LIG:B")
    return best_chain, resseq, resname



class LigandSASA(Step):
    def __init__(self, input_dir = None,  output_dir: str = ''):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)     

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['best_structure']
            substrate_smiles = row['substrate_smiles']
            pdb_file = Path(self.input_dir / f"{best_structure_name}.pdb")
            row_result = {}

            # define default results
            default_result = {
                'sasa_ligand_in_complex': None,
                'sasa_ligand_alone': None,
                'buried_sasa': None,
                'percentage_buried_sasa': None}

            print(f"Processing PDB file: {pdb_file.name}")
            
            try:
                ligand = select_ligand_from_smiles_via_composition(pdb_file, substrate_smiles)
                if not ligand:
                    raise RuntimeError("No ligand chains found or composition match failed.")
                chain_id, resseq, resname = ligand
                #print(ligand)

                # Extract ligand from PDB file containing docked protein-ligand structure and save in temporary directory
                with TemporaryDirectory() as tmpdir:
                    ligand_path = Path(tmpdir) / "ligand.pdb"

                    io = PDBIO()
                    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_file))[0]
                    io.set_structure(structure)
                    io.save(str(ligand_path), select=SingleLigandSelect(chain_id, resseq, resname))

                    # load complex & ligand into FreeSASA
                    structure_complex = freesasa.Structure(str(pdb_file), options={'hetatm': True})
                    structure_ligand = freesasa.Structure(str(ligand_path), options={'hetatm': True})

                # Run SASA calculation
                result_ligand = freesasa.calc(structure_ligand)
                result_complex = freesasa.calc(structure_complex)

                # Get SASA values
                selection = [f"ligand, chain {chain_id} and resn {resname} and resi {resseq}"]
                sasa_ligand_in_complex = freesasa.selectArea(selection, structure_complex, result_complex)
                sasa_ligand_alone = result_ligand.totalArea()

                # Buried SASA = exposed alone - exposed in complex
                buried_sasa = sasa_ligand_alone - sasa_ligand_in_complex["ligand"]
                if sasa_ligand_alone > 0:
                    percent_buried = (buried_sasa / sasa_ligand_alone) * 100
                else:
                    percent_buried = 0.0

                row_result['sasa_ligand_in_complex'] = sasa_ligand_in_complex["ligand"]
                row_result['sasa_ligand_alone'] = sasa_ligand_alone
                row_result['buried_sasa'] = buried_sasa
                row_result['percentage_buried_sasa'] = percent_buried

            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")
                row_result.update(default_result)

            results.append(row_result)
        return results
                    

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            print("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)        
        results_df = pd.DataFrame(results) # Convert list of row-dictionaries to df       
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1) # Merge with input df

        return output_df
