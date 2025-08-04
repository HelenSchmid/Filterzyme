import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn


def argparser():
    parser = argparse.ArgumentParser(description='Infer active site residues')
    parser.add_argument('-i', '--fasta_file', required=True, help='input fasta of ALL sequences to be infered')
    parser.add_argument('-m', '--model', default='/scratch/user/uqwriege/masters_thesis/ASTformer/tests/gym_tanh_bidirectional_2/models/07-06-24_16-09_128_2_0.1_32_best_model.pth',required=True, help='model to be used for inference')
    parser.add_argument('-b', '--batch_size', type=int , default=100, help='to minimise RAM usage, batch size can be adjusted (i.e. how many seqs to infer in each round)')
    parser.add_argument('-o', '--output_dir', required=True, help='output directory')
    return parser.parse_args()


class ProteinLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(ProteinLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes protein embeddings as inputs and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        # The linear layer that maps from hidden state space to the output space
        self.hidden2out = nn.Linear(hidden_dim*2, output_dim)
        
        self.best_model_path = ""
        
    def forward(self, x):
        x = torch.tanh(x)
        lstm_out, _ = self.lstm(x)
        output = self.hidden2out(lstm_out)
        return output
    

def load_fasta(fasta_file):
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(record)
    return seqs


def manual_pad_sequence_tensors(tensors, target_length, padding_value=0):
    padded_tensors = []
    for tensor in tensors:
        # Check if padding is needed along the first dimension
        if tensor.size(0) < target_length:
            pad_size = target_length - tensor.size(0)
            # Create a padding tensor with the specified value
            padding_tensor = torch.full((pad_size, tensor.size(1)), padding_value, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the padding tensor to the original tensor along the first dimension
            padded_tensor = torch.cat([tensor, padding_tensor])
        # If the tensor is longer than the target length, trim it along the first dimension
        else:
            padded_tensor = tensor[:target_length, :]
        padded_tensors.append(padded_tensor)
    return padded_tensors


def main():
    args = argparser()

    # if output dir does not exist, create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # create a subdirectory in the output dir for the embeddings
    embedding_dir = Path(args.output_dir) / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # create a subdirectory in the batch fasta files
    batch_dir = Path(args.output_dir) / "batch_fasta_files"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # load the seqs
    sequences = load_fasta(args.fasta_file)
    
    # remove any sequences with length > 900
    sequences = [seq for seq in sequences if len(seq.seq) <= 900]
    
    # seperate the sequences into batches
    batch_size = args.batch_size
    
    if batch_size > len(sequences):
        num_batches = 1
        handle_last_batch = True
    else:
        num_batches = len(sequences) // batch_size
        if len(sequences) % batch_size != 0:
            num_batches += 1
            handle_last_batch = True
    
    # create a fasta file for each batch in batch_dir
    for i in range(num_batches):
        with open(batch_dir / f"batch_{i}.fasta", "w") as f:
            if i == num_batches - 1 and handle_last_batch:
                for seq in sequences[i*batch_size:]:
                    f.write(f">{seq.id}\n{seq.seq}\n")
            else:
                for seq in sequences[i*batch_size:(i+1)*batch_size]:
                    f.write(f">{seq.id}\n{seq.seq}\n")
    
    # get list of batch fasta files
    batch_files = [str(file) for file in batch_dir.iterdir()]
    
    # start the inference
    for fasta in tqdm(batch_files):
        # ______________________________________________________________________________________________________________
        # STEP 1: Get the embeddings and store temporarily in embedding_dir... will be cleaned after each batch inferred
        # ______________________________________________________________________________________________________________
        
        # Need to CLEAN! referring toe extract.py this way is super stupid. BLEUGH fix later
        print('Extracting Embeddings!')
        command = f'python extract.py esm2_t36_3B_UR50D {fasta} {embedding_dir} --include per_tok'
        # Execute the command and capture the exit status
        os.environ["MKL_THREADING_LAYER"] = "GNU"                 # had to specify cause was getting weird errors otherwise... needs to be fixed
        os.system(command)
    
        
        # __________________________________________________________________ 
        # STEP 2: Load the embeddings and process into tensors for inference
        # __________________________________________________________________ 
        
        # Load the embeddings
        model_max_length = 900
        embeddings_tensors = []
        embeddings_labels = []
        for file in Path(embedding_dir).iterdir():
            embedding_file = torch.load(file)
            tensor = embedding_file['representations'][36] # have to get the last layer (36) of the embeddings... very dependant on ESM model used! 36 for medium ESM2
            label = embedding_file['label']
            embeddings_tensors.append(tensor)
            embeddings_labels.append(label)
            # delete the file after loading
            os.remove(file)
        
        # clean up the labels to just get accession code
        embedding_UPaccessions = [label.split("|")[1] for label in embeddings_labels]
        del embeddings_labels
        
        # pad the tensors
        padded_tensor = manual_pad_sequence_tensors(embeddings_tensors, model_max_length)
        del embeddings_tensors
        padded_tensor = torch.stack(padded_tensor)
                
        if padded_tensor.shape[1] != model_max_length:
            print(padded_tensor.shape)
            raise ValueError("The padded tensor is not of the expected length, probably you gave it sequences which are >900 in length")
        
        loader = DataLoader(padded_tensor, batch_size=100, shuffle=False)
        
        # _______________________________________________________ 
        # STEP 3: Load and infer with model 
        # _______________________________________________________ 
        
        print('Predicting active sites...')
        
        model = ProteinLSTM(embedding_dim=2560, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
        model.load_state_dict(torch.load(args.model))
        
        model.eval()
        with torch.no_grad():  # Disables gradient calculation
            for inputs in loader:
                outputs = model(inputs)
                predicted = torch.sigmoid(outputs).round()  # Convert logits to probabilities and then to binary predictions
        
        predictions = []
        # get the predictions from each row in predicted tensor with 3D shape, row is 1st dimension
        for i in range(predicted.shape[0]):
            predictions.append(torch.where(predicted[i] == 1)[0].tolist())
        
        print('DONE')
        
        # _______________________________________________________
        # STEP 4: Clean up and save the predictions
        # _______________________________________________________
        
        # get seqs in the same order of embedding_UPaccessions
        seqs = []
        for id in embedding_UPaccessions:
            for seq in sequences:
                if id in seq.id:
                    seqs.append(seq)
                    break
        
        # get the corresponding residues from the active site positions
        active_site_residues = []
        for i, positions in enumerate(predictions):
            residues = []
            seq = seqs[i]
            for position in positions:
                residues.append(seq[position])
            active_site_residues.append(residues)
            
        # make the output positions and residues in the following format, D|S|H, 10|20|1 etc
        predictions = ["|".join([str(x) for x in positions]) for positions in predictions]
        active_site_residues = ["|".join(x for x in residues) for residues in active_site_residues]
        
        # Create a dataframe with embedding_UniProtAccessions, active_site_residues, and active_site_positions
        df = pd.DataFrame({"Entry": embedding_UPaccessions, "active_site_residues": active_site_residues, "active_site_positions": predictions})
        
        # save the dataframe to a tsv file in the output dir
        df.to_csv(Path(args.output_dir) / f"{fasta.split('/')[-1].split('.')[0]}_AS_predictions.tsv", sep="\t", index=False)
    
    # _______________________________________________________
    # STEP 5: Consolidate all the batch tsv files into one
    # _______________________________________________________
    
    # delete all files in the batch_dir
    for file in Path(batch_dir).iterdir():
        os.remove(file)
    
    # delete the batch_dir
    os.rmdir(batch_dir)
    
    # delete the embeddings dir
    os.rmdir(embedding_dir)
    
    # After inference, merge all the tsv files into one
    tsv_files = [str(file) for file in Path(args.output_dir).iterdir() if file.suffix == ".tsv"]
    
    # final df
    final_df = pd.concat([pd.read_csv(file, sep="\t") for file in tsv_files])
    
    # remove the batch tsvs
    for file in tsv_files:
        os.remove(file)
    
    # save that mf
    final_df.to_csv(Path(args.output_dir) / "final_AS_predictions.tsv", sep="\t", index=False)
        

if __name__ == '__main__':
    main()