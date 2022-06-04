import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, SAGEConv, DNAConv, ARMAConv, ChebConv, GINConv, GatedGraphConv, SplineConv, TopKPooling, GATConv, EdgePooling, TAGConv,DynamicEdgeConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool


class PhpNet(nn.Module):
    def __init__(self, EMBED_SIZE, EMBED_DIM, LSTM_SIZE,
                 LSTM_LAYER, LSTM_BIDIR, FC_DROP, FC_OUT):
        super(PhpNet, self).__init__()

        self.embed = nn.Embedding(num_embeddings=EMBED_SIZE,
                                  embedding_dim=EMBED_DIM)
        # self.multi = nn.MultiheadAttention(EMBED_DIM, 5)
        self.lstm1 = nn.GRU(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        # self.dropout = nn.Dropout(FC_DROP)

        if LSTM_BIDIR is True:
            self.fc1 = nn.Linear(LSTM_SIZE * 3 * 2, 200)
            self.fc2 = nn.Linear(200, 50)
            self.fc3 = nn.Linear(50, FC_OUT)
        else:
            self.fc1 = nn.Linear(LSTM_SIZE, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, FC_OUT)


    def forward(self, x):
        x = self.embed(x)
        # x, _ = self.multi(x, x, x)
        output, (hidden) = self.lstm1(x)
        # x = torch.cat((output[:, 0, :], output[:, -1, :]), dim=1)
        x = torch.cat((hidden[0, :, :], hidden[1, :, :], hidden[2, :, :], hidden[-3, :, :], hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class PhpNetW2V(nn.Module):
    def __init__(self, EMBED_DIM, LSTM_SIZE,
                 LSTM_LAYER, LSTM_BIDIR, FC_DROP, FC_OUT):
        super(PhpNetW2V, self).__init__()

        # self.multi = nn.MultiheadAttention(EMBED_DIM, 5)
        self.lstm1 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        # self.dropout = nn.Dropout(FC_DROP)

        self.fc1 = nn.Linear(LSTM_SIZE * 2 * 2, 10)
        self.fc2 = nn.Linear(10, 4)


    def forward(self, x):
        # x, _ = self.multi(x, x, x)
        output, (hidden) = self.lstm1(x)
        x = torch.cat((output[:, 0, :], output[:, -1, :]), dim=1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = nn.Softmax()(self.fc2(x))
        return (x)

class PhpNet2(nn.Module):
    def __init__(self, EMBED_SIZE, EMBED_DIM, LSTM_SIZE,
                 LSTM_LAYER, LSTM_BIDIR, FC_DROP, FC_OUT):
        super(PhpNet2, self).__init__()

        self.embed = nn.Embedding(num_embeddings=EMBED_SIZE,
                                  embedding_dim=EMBED_DIM)
        self.lstm11 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        self.lstm12 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        self.lstm13 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        self.lstm14 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)

        self.lstm2 = nn.LSTM(input_size=EMBED_DIM,
                             hidden_size=LSTM_SIZE,
                             num_layers=LSTM_LAYER,
                             batch_first=True,
                             bidirectional=LSTM_BIDIR)
        self.dropout = nn.Dropout(FC_DROP)

        if LSTM_BIDIR is True:
            self.fc1 = nn.Linear(800, 400) #LSTM_SIZE * 2 * 2
            self.fc2 = nn.Linear(400, 100)
            self.fc3 = nn.Linear(100, FC_OUT)
        else:
            self.fc1 = nn.Linear(LSTM_SIZE, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, FC_OUT)

    def forward(self, x):
        x = self.embed(x)
        # x, _ = self.multi(x, x, x)
        length = len(x[0])
        x1,x2,x3,x4 = torch.split(x,length//4, dim=1)
        output11, (hidden11,_) = self.lstm11(x1)
        output12, (hidden12,_) = self.lstm12(torch.cat((x1,x2),dim=1))
        output13, (hidden13,_) = self.lstm13(torch.cat((x1,x2,x3),dim=1))
        output14, (hidden14,_) = self.lstm14(torch.cat((x1,x2,x3,x4),dim=1))
        x = torch.cat((output11[:,-1,:],output12[:,-1,:],output13[:,-1,:],output14[:,-1,:]),dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.Softmax()(self.fc3(x))
        return (x)

class PhpNetGraph(nn.Module):
    def __init__(self):
        super(PhpNetGraph, self).__init__()

        self.conv1 = GCNConv(107, 200)
        self.pool1 = EdgePooling(200,edge_score_method=EdgePooling.compute_edge_score_softmax,dropout=0.2, add_to_edge_score=0.1)
        self.conv2 = GCNConv(200, 400)
        self.pool2 = EdgePooling(400, edge_score_method=EdgePooling.compute_edge_score_softmax,dropout=0.2, add_to_edge_score=0.1)
        self.conv3 = GCNConv(400, 600)
        self.pool3 = EdgePooling(600,edge_score_method=EdgePooling.compute_edge_score_softmax,dropout=0.2, add_to_edge_score=0.1)
        # self.conv4 = GCNConv(600, 800)
        # self.pool4 = EdgePooling(800)
        # self.conv5 = GCNConv(800, 1000)
        # self.pool5 = EdgePooling(1000)

        self.lin1 = nn.Linear(600, 200)
        self.lin2 = nn.Linear(200, 20)
        self.lin3 = nn.Linear(20,4)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=data.batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training,p=0.5)

        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        #
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)

        # x = self.conv4(x, edge_index)
        # x, edge_index, batch, _ = self.pool4(x, edge_index, batch=batch)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.5)
        #
        # x = self.conv5(x, edge_index)
        # x, edge_index, batch, _= self.pool5(x, edge_index, batch=batch)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.5)

        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(self.lin3(x))
        return (nn.Softmax()(x))


class PhpNetGraphTokens(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokens, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=200)

        self.conv1 = GCNConv(4000, 6000)
        self.pool1 = EdgePooling(6000)
        self.conv2 = GCNConv(6000, 6000)
        self.pool2 = EdgePooling(6000)
        self.conv3 = GCNConv(6000, 6000)
        self.pool3 = EdgePooling(6000)



        self.lin1 = nn.Linear(6000, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin3 = nn.Linear(500, 2)
        self.lin4 = nn.Linear(100,4)

    def forward(self, dataGraph, dataTokens=None):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)

        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)

        # # x = F.dropout(x, training=self.training,p=0.5)
        #
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)
        #
        # # # x = F.leaky_relu(x,0.2)
        x = F.dropout(x, training=self.training, p=0.5)
        # #
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x

class PhpNetGraphTokensAST(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensAST, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=200)
        self.conv1 = GCNConv(200,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 2000)
        self.pool2 = EdgePooling(2000)
        self.conv3 = GCNConv(2000, 2000)
        self.pool3 = EdgePooling(2000)
        self.conv4 = GCNConv(6000, 8000)
        self.pool4 = EdgePooling(8000)

        self.lin1 = nn.Linear(2000, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)

    def forward(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)
        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.3)

        # # x = F.dropout(x, training=self.training,p=0.5)
        #
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)

        # # x = F.leaky_relu(x,0.2)
        # # x = F.dropout(x, training=self.training, p=0.5)
        #
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.conv1 = GCNConv(2000,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        self.conv3 = GCNConv(4000, 4000)
        self.pool3 = EdgePooling(4000)
        self.conv4 = GCNConv(2000, 2000)
        self.pool4 = EdgePooling(8000)

        #
        self.embed = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=200,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)



        self.lin1 = nn.Linear(5200, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin3 = nn.Linear(200, 100)
        self.lin4 = nn.Linear(100,4)

    def forward(self, dataGraph, dataTokens):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)
        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.3)

        # # x = F.dropout(x, training=self.training,p=0.5)
        #
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)

        # # x = F.leaky_relu(x,0.2)
        # # x = F.dropout(x, training=self.training, p=0.5)
        #
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)


        x = global_max_pool(x, batch)


        x1 = self.embed(dataTokens)
        output1, (hidden1) = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :],hidden1[1, :, :],hidden1[2, :, :],hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)
        # x1 = F.relu(self.fc1(x1))
        # x1 = F.relu(self.fc2(x1))
        x = torch.cat([x,x1], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x



class PhpNetGraphDependenceTokens(nn.Module):
    def __init__(self):
        super(PhpNetGraphDependenceTokens, self).__init__()

        self.embed1 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=100)
        self.conv11 = GCNConv(2000, 2000)
        self.pool11 = EdgePooling(2000,edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv21 = GCNConv(2000, 2000)
        self.pool21 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv31 = GCNConv(2000, 2000)
        self.pool31 = EdgePooling(2000,edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv41 = GCNConv(2000, 2000)
        self.pool41 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)

        self.embed2 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=100)
        self.conv12 = GCNConv(2000, 2000)
        self.pool12 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv22 = GCNConv(2000, 2000)
        self.pool22 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv32 = GCNConv(2000, 2000)
        self.pool32 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)
        self.conv42 = GCNConv(2000, 2000)
        self.pool42 = EdgePooling(2000, edge_score_method=EdgePooling.compute_edge_score_softmax)


        self.fc1 = nn.Linear(400, 300)


        self.lin01 = nn.Sequential(nn.Linear(4000,3000), nn.BatchNorm1d(3000))
        self.lin02 = nn.Sequential(nn.Linear(3000, 1600), nn.BatchNorm1d(1600))
        self.lin1 = nn.Sequential(nn.Linear(1600, 4))
        self.lin2 = nn.Sequential(nn.Linear(750, 400), nn.BatchNorm1d(400))
        self.lin3 = nn.Sequential(nn.Linear(400,100), nn.BatchNorm1d(100))
        self.lin4 = nn.Linear(100, 4)

    def forward(self, dataProgramDependence, dataControlFlow, dataTokens):


        x1, edge_index = dataProgramDependence.x.long(), dataProgramDependence.edge_index

        # dependence
        pre_x_len = len(x1)
        x1 = self.embed1(x1)
        x1 = x1.reshape(pre_x_len, -1)

        x1 = self.conv11(x1, edge_index)
        x1, edge_index, batch,_ = self.pool11(x1, edge_index, batch=dataProgramDependence.batch)
        x1 = F.relu(x1)
        # x1 = F.dropout(x1, training=self.training,p=0.5)

        x1 = self.conv21(x1, edge_index)
        x1, edge_index, batch, _= self.pool21(x1,edge_index,batch=batch)
        x1 = F.relu(x1)
        # x1 = F.dropout(x1, training=self.training, p=0.5)
        #
        x1 = self.conv31(x1, edge_index)
        x1, edge_index, batch, _= self.pool31(x1, edge_index, batch=batch)
        x1 = F.relu(x1)
        # x1 = F.dropout(x1, training=self.training, p=0.5)
        x1 = self.conv41(x1, edge_index)
        x1, edge_index, batch, _ = self.pool41(x1, edge_index, batch=batch)
        x1 = F.relu(x1)

        x1 = global_max_pool(x1, batch)

        #cfg
        x2, edge_index = dataControlFlow.x.long(), dataControlFlow.edge_index

        pre_x_len = len(x2)
        x2 = self.embed2(x2)
        x2 = x2.reshape(pre_x_len, -1)

        x2 = self.conv12(x2, edge_index)
        x2, edge_index, batch, _ = self.pool12(x2, edge_index, batch=dataControlFlow.batch)
        x2 = F.relu(x2)
        # x2 = F.dropout(x2, training=self.training, p=0.5)

        x2 = self.conv22(x2, edge_index)
        x2, edge_index, batch, _ = self.pool22(x2, edge_index, batch=batch)
        x2 = F.relu(x2)
        # x2 = F.dropout(x2, training=self.training, p=0.5)
        #
        x2 = self.conv32(x2, edge_index)
        x2, edge_index, batch, _ = self.pool32(x2, edge_index, batch=batch)
        x2 = F.relu(x2)
        # x2 = F.dropout(x2, training=self.training, p=0.5)
        x2 = self.conv42(x2, edge_index)
        x2, edge_index, batch, _ = self.pool42(x2, edge_index, batch=batch)
        x2 = F.relu(x2)

        x2 = global_max_pool(x2, batch)


        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.lin01(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin02(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin1(x))
        return x


class PhpNetASTGraphDependenceTokens(nn.Module):
    def __init__(self):
        super(PhpNetASTGraphDependenceTokens, self).__init__()

        #dependence
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv11 = GatedGraphConv(400, 3)

        #AST
        self.embed3 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv12 = GatedGraphConv(400, 3)

        # cfg
        self.embed2 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv14 = GatedGraphConv(400, 3)

        self.lin1 = nn.Linear(1200, 1000)
        self.lin2 = nn.Linear(1000, 750)
        self.lin21 = nn.Linear(750, 500)
        self.lin22 = nn.Linear(500, 400)
        self.lin3 = nn.Linear(400,100)
        self.lin4 = nn.Linear(100, 4)

    def forward(self, dataProgramDependence, dataAST, dataControlFlow2):


        x1, edge_index = dataProgramDependence.x.long(), dataProgramDependence.edge_index

        # dependence
        pre_x_len = len(x1)
        x1 = self.embed1(x1)
        x1 = x1.reshape(pre_x_len, -1)

        x1 = self.conv11(x1, edge_index)

        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training, p=0.1)

        x1 = global_max_pool(x1, dataProgramDependence.batch)

        #ast
        x2, edge_index = dataAST.x.long(), dataAST.edge_index
        pre_x_len = len(x2)
        x2 = self.embed3(x2)
        x2= x2.reshape(pre_x_len, -1)

        x2 = self.conv12(x2, edge_index)

        x2 = F.relu(x2)
        x2 = F.dropout(x2, training=self.training, p=0.1)

        x2 = global_max_pool(x2, dataAST.batch)


        #cfg embed
        x4, edge_index = dataControlFlow2.x.long(), dataControlFlow2.edge_index
        pre_x_len = len(x4)
        x4 = self.embed2(x4)
        x4 = x4.reshape(pre_x_len, -1)

        x4 = self.conv14(x4, edge_index)
        x4 = F.relu(x4)
        x4 = F.dropout(x4, training=self.training, p=0.1)

        x4 = global_max_pool(x4, dataControlFlow2.batch)


        x = torch.cat([x1,x2,x4], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.lin21(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.lin22(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.lin3(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.lin4(x))
        return x



class PhpNetGraphTokensSingle(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensSingle, self).__init__()

        self.embed = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=100)
        self.conv14 = GatedGraphConv(2000, 4)

        self.conv1 = nn.Conv1d(1, 50, 3)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(50, 100, 3)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(100, 150, 3)
        self.pool3 = nn.MaxPool1d(3)
        # self.conv4 = nn.Conv1d(50, 100, 5)
        # self.pool4 = nn.MaxPool1d(2)

        self.lin1 = nn.Linear(10950, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin21 = nn.Linear(500, 4)

    def forward(self, dataIn):



        #cfg embed
        x, edge_index = dataIn.x.long(), dataIn.edge_index
        pre_x_len = len(x)
        x = self.embed(x)
        x = x.reshape(pre_x_len, -1)

        x = self.conv14(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.5)

        x = global_max_pool(x, dataIn.batch)
        x = x.reshape(len(x),1,-1)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # x = F.dropout(x, training=self.training, p=0.5)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = F.dropout(x, training=self.training, p=0.5)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)

        x = x.reshape(len(x),-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, training=self.training, p=0.5)
        # x = F.relu(self.lin21(x))
        return x

class PhpNetDependenceTokensCFGComb(nn.Module):
    def __init__(self):
        super(PhpNetDependenceTokensCFGComb, self).__init__()

        self.embed1 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv11 = GatedGraphConv(500, 3)

        self.embed2 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv12 = GatedGraphConv(500, 3)

        self.embed3 = nn.Embedding(num_embeddings=5000,
                                   embedding_dim=20)
        self.conv13 = GatedGraphConv(500, 3)

        self.conv1 = nn.Conv1d(1, 20, 5)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(20, 50, 5)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(50, 100, 5)
        self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(50, 100, 5)
        # self.pool4 = nn.MaxPool1d(2)

        self.lin1 = nn.Linear(2200, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin21 = nn.Linear(500, 4)

    def forward(self, dataDep, dataCFG, dataAST):



        #dep embed
        x1, edge_index = dataDep.x.long(), dataDep.edge_index
        pre_x_len = len(x1)
        x1 = self.embed1(x1)
        x1 = x1.reshape(pre_x_len, -1)

        x1 = self.conv11(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training, p=0.5)

        x1 = global_max_pool(x1, dataDep.batch)

        # cfg embed
        x2, edge_index = dataCFG.x.long(), dataCFG.edge_index
        pre_x_len = len(x2)
        x2 = self.embed2(x2)
        x2 = x2.reshape(pre_x_len, -1)

        x2 = self.conv12(x2, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, training=self.training, p=0.5)

        x2 = global_max_pool(x2, dataCFG.batch)

        # ast embed
        x3, edge_index = dataAST.x.long(), dataAST.edge_index
        pre_x_len = len(x3)
        x3 = self.embed3(x3)
        x3 = x3.reshape(pre_x_len, -1)

        x3 = self.conv13(x3, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, training=self.training, p=0.5)

        x3 = global_max_pool(x3, dataAST.batch)

        x = torch.cat([x1, x2, x3], dim=1)

        x = x.reshape(len(x),1,-1)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)

        x = x.reshape(len(x),-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, training=self.training, p=0.5)
        # x = F.relu(self.lin21(x))
        return x

class PhpNetGraphTokensCombineFileLevel(nn.Module):
    def __init__(self, VOCAB_SIZE_graph, VOCAB_SIZE_tokens):
        super(PhpNetGraphTokensCombineFileLevel, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=VOCAB_SIZE_graph,
                                  embedding_dim=100)
        self.conv1 = GCNConv(2000,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        self.conv3 = GCNConv(4000, 4000)
        self.pool3 = EdgePooling(4000)

        self.embed = nn.Embedding(num_embeddings=VOCAB_SIZE_tokens,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=200,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)

        self.lin1 = nn.Linear(5200, 1000)
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 4)

    def forward(self, dataGraph, dataTokens, eval=False):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)
        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x1 = self.embed(dataTokens)
        output1, (hidden1) = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :],hidden1[1, :, :],hidden1[2, :, :],hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)
        x = torch.cat([x,x1], dim=1)

        x = F.relu(self.lin1(x))
        if eval is False:
            print('masuk')
            x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        if eval is False:
            x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin3(x))
        return(x)