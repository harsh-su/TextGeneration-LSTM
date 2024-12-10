#defineing model 
import math
import torch
import torch.nn as nn
import torch.optim as optim

class CustomLSTM(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #defening linear layers for each gate 
        self.input_gate = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.forget_gate = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.cell_gate = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.output_gate = nn.Linear(input_dim+hidden_dim,hidden_dim)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 /math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)

    def forward(self,x,init_states=None):
        """Assume x has shape(batcb,sequnce,features)"""
        batch_size,seq_len,_ = x.size()
        hidden_seq=[]
        if init_states is None:
            hiddden_state ,cell_state =(torch.zeros(batch_size,self.hidden_dim).to(x.device),
                                        torch.zeros(batch_size,self.hidden_dim).to(x.device))
        else :
            hiddden_state,cell_state = init_states
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden states for each gate to use as input 
            combined_input = torch.cat((x_t,hiddden_state),dim=1)

            #applying Linear transformation 

            i_t = torch.sigmoid(self.input_gate(combined_input))
            f_t = torch.sigmoid(self.forget_gate(combined_input))
            # Potential New memory at t 
            g_t = torch.tanh(self.cell_gate(combined_input)) 
            o_t = torch.sigmoid(self.output_gate(combined_input))

            #updating Cell and hidden states 

            cell_state = f_t * cell_state + i_t * g_t
            hiddden_state = o_t * torch.tanh(cell_state)

            hidden_seq.append(hiddden_state.unsqueeze(0))
        
        # reshape hidden_seq 
        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()

        return hidden_seq, (hiddden_state,cell_state)

class nxtword(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,128)
        self.lstm = CustomLSTM(128,128)       
        self.fc1 = nn.Linear(128,vocab_size)

    def forward(self,x):
        #creating embedding from input 
        embedded_x = self.embedding(x)
        #passing embeding to lstm 
        lstm_out, (h_n,c_n) =self.lstm(embedded_x)
        #captuing the last  hidden state
        last_hidden_state = lstm_out[:,-1,:]
        # Pass to last fc layer 
        output = self.fc1(last_hidden_state)
        
        return output