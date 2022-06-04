import pandas as pd
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from pathlib import Path

BATCH_SIZE=1
INDEX_TOKEN_FOR_SEMI_IN_GRAPH_ENCODER=251

#### NEED to add these, and prepare it
#all_pred=[]
#all_lbl=[]
#all_codes=[]
#all_graphs=[]
#sensitivity_grad=[]
#sensitivity_grad_graph=[]
#all_filenames=[]

#all_grad=[]
#all_grad_graph=[]


def set_eval_zerodrop(model):
    model.zero_grad()
    model.train()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
            module.training = False

        elif isinstance(module, torch.nn.GRU):
            module.dropout = 0
            
    return(model)
    
def get_graph_tokens(graph_nodes):
    temp_graph = []
    temp_glen=[]
    for node in graph_nodes:
        k=node.tolist()
        glen = len(np.nonzero(node.tolist())[0])
        temp_graph+=k[0:glen]
        temp_glen.append(glen)
    return(temp_graph,temp_glen)
    
def get_weights_grads_from_model(model,pred,embeddings_name):
    try:
        pred.max().backward()
    except:
        ## Backprop already ran
        None
        
    for name, parameter in model.named_parameters():

        ## Code tokens
        if embeddings_name+'.weight' in name:
            ## Weights 
            weight_embed = parameter
            #print(weight_embed)

            ## Gradient activations
            grad_embed = parameter.grad

    sensitivity_weight = weight_embed.mean(dim=1).tolist()
    sensitivity_grad = grad_embed.mean(dim=1).abs().tolist()
    
    ## Weights only once because its never updated due to no backprop step update
    ## Gradient different for each batch, thus for exclusive insights for each sample, 
    ## set the batchsize equal to 1.
    ## Gradient can either be positive or negative, but the magnitude is important,
    ## thus, absolute value-ing the tensor is acceptable
    
    return(sensitivity_weight,sensitivity_grad,grad_embed.to(torch.device('cpu')).detach().numpy())

    
def rescale(sensitivity_grad):
    for i in range(len(sensitivity_grad)):
        #sensitivity_grad[i] = np.interp(sensitivity_grad[i],
        #                                (min(sensitivity_grad[i]), max(sensitivity_grad[i])),
        #                                (0, 1))
        sensitivity_grad[i] = minmax_scale(sensitivity_grad[i])
        
    return(sensitivity_grad)

    
def color_dict_g(act):
    if act == 0.0:
        return('255,255,255')
    if act > 0.0 and act < 0.1:
        return('255,255,255')
    elif act >= 0.1 and act < 0.2:
        return('0,0,255')
    elif act >= 0.2 and act < 0.3:
        return('0,206,209')
    elif act >= 0.3 and act < 0.4:
        return('0,128,0')
    elif act >= 0.4 and act < 0.5:
        return('153,255,204')
        #return('160,82,45')
    elif act >= 0.5 and act < 0.6:
        return('95,158,160')
    elif act >= 0.6 and act < 0.7:
        return('255,128,0')
    elif act >= 0.7 and act < 0.8:
        return('255,0,255')
    elif act >= 0.8 and act < 0.9:
        return('255,0,0')
    else:
        return('128,0,0')

### Optional
def getClass(all_pred):
    def getIndexClass(x):
        return(x.index(max(x)))
    
    probs = pd.Series(all_pred)
    all_predicted = probs.apply(getIndexClass)
    return(all_predicted)


def get_tokens_grads(all_codes,sensitivity_weight,sensitivity_grad,encoder):
    all_tokens = []
    activations_weights = []
    activations_grads = []

    ### Get the token for each function and it specific weights and gradients in correct order
    if BATCH_SIZE == 1:
        for i in range(len(all_codes)):
            all_tokens.append([encoder.index_to_token[tok] for tok in all_codes[i]])
            activations_weights.append([sensitivity_weight[tok] for tok in all_codes[i]])
            activations_grads.append([sensitivity_grad[i][tok] for tok in all_codes[i]])
    else:
        loopflag=True
        for i in range(len(all_codes)):
            all_tokens.append([encoder.index_to_token[tok] for tok in all_codes[i]])
            activations_weights.append([sensitivity_weight[tok] for tok in all_codes[i]])
        k=0
        for i in range(len(sensitivity_grad)):
            try:
                for batchloop in range(BATCH_SIZE):
                    activations_grads.append([sensitivity_grad[i][tok] for tok in all_codes[k]])
                    k+=1
            except:
                None
                
    return(all_tokens,activations_weights,activations_grads)
                
def create_html(all_tokens,all_predicted,activations_weights,activations_grads,integrated_grads,all_lbl,all_filenames,*args):
    html_list_grads=[]
    html_list_ig=[]
    
    len_args = len(args)

    for i in range(len(all_tokens)):
        html_content_grads = '<h1 style="text-align: center;"><span style="text-decoration: underline;"><strong>Saliency Map</strong></span></h1>'
        html_content_grads += 'Label: %d <br /> Predicted: %d <br /> Number of UNKNOWN tokens: %d <br />' % (all_lbl[i], all_predicted[i], all_tokens[i].count('<unk>'))
        html_content_grads += 'Distribution of activations: '

        html_content_ig = '<h1 style="text-align: center;"><span style="text-decoration: underline;"><strong>Integrated Gradients</strong></span></h1>'
        html_content_ig += 'Label: %d <br /> Predicted: %d <br /> Number of UNKNOWN tokens: %d <br />' % (all_lbl[i], all_predicted[i], all_tokens[i].count('<unk>'))
        html_content_ig += 'Distribution of activations: '
        
        for k in np.arange(0,1,0.1):
            html_content_grads += '<font style="background: rgba(%s, 0.5)">%.1f</font> '%(color_dict_g(k),k)
            html_content_ig += '<font style="background: rgba(%s, 0.5)">%.1f</font> '%(color_dict_g(k),k)

        html_content_grads += '<br /> <br />'
        html_content_ig += '<br /> <br />'
        
        
        wcount=0
        statement=0
        
        for word, alpha_g in zip(all_tokens[i], activations_grads[i]):
            if not word == '<pad>':
                wcount+=1
                if word == '<unk>':
                    html_content_grads += '<font style="background: rgba(%s, %.3f)">UNK</font>\n' % (color_dict_g(alpha_g), 0.5)
                    
                else:
                    html_content_grads += '<font style="background: rgba(%s, %.3f)">%s</font>\n' % (color_dict_g(alpha_g), 0.5, word)
                    if word ==';' or word =='SEMI':
                        html_content_grads += '<br />\n'
                    if len_args!=0:
                        if wcount==args[0][i][statement]:
                            html_content_grads += '<br />\n'
                            statement+=1
                            wcount=0
        
        html_list_grads.append(html_content_grads)
        
        wcount=0
        statement=0
        
        for word, alpha_g in zip(all_tokens[i], integrated_grads[i]):
            if not word == '<pad>':
                wcount+=1
                if word == '<unk>':
                    html_content_ig += '<font style="background: rgba(%s, %.3f)">UNK</font>\n' % (color_dict_g(alpha_g), 0.5)
                    
                else:
                    html_content_ig += '<font style="background: rgba(%s, %.3f)">%s</font>\n' % (color_dict_g(alpha_g), 0.5, word)
                    if word ==';' or word =='SEMI':
                        html_content_ig += '<br />\n'
                    if len_args!=0:
                        if wcount==args[0][i][statement]:
                            html_content_ig += '<br />\n'
                            statement+=1
                            wcount=0
            
        html_list_ig.append(html_content_ig)

    mydf = pd.DataFrame({'function': all_tokens,
                         'weights': activations_weights,
                         'gradients': activations_grads,
                         'ig': integrated_grads,
                         'html_grads': html_list_grads,
                         'html_ig': html_list_ig,
                         'label': all_lbl,
                         'predicted': all_predicted,
                         'filename': all_filenames})
    return(mydf)

def integrated_gradients(inp,model,device,graph,embed,steps=50):
    scaled_grads=[]
    
    if graph == False:
        graph_or_token=inp[1]
        baseline = torch.LongTensor([np.zeros(len(inp[1][0]))]).to(device)
        scaled_inputs = [torch.round(baseline + (float(i)/steps)*(inp[1]-baseline)).type(torch.LongTensor).to(device) for i in range(0, steps+1)]
        for sinp in scaled_inputs:
            pred = model(inp[0],sinp)
            sensitivity_weight,sensitivity_grad_batch,grad = get_weights_grads_from_model(model,pred,embed)
            scaled_grads.append(sensitivity_grad_batch)
    else:
        graph_or_token=inp[0].x
        baseline=torch.zeros_like(inp[0].x)
        scaled_x = [torch.round(baseline + (float(i)/steps)*(inp[0].x-baseline)).type(torch.LongTensor).to(device) for i in range(0, steps+1)]
        scaled_inputs=[]
        for x in scaled_x:
            single_x = inp[0].clone() 
            single_x.x = x
            scaled_inputs.append(single_x)
            
        for sinp in scaled_inputs:
            pred = model(sinp,inp[1])
            sensitivity_weight,sensitivity_grad_batch,grad = get_weights_grads_from_model(model,pred,embed)
            scaled_grads.append(sensitivity_grad_batch)
    
    scaled_grads = np.divide((scaled_grads[:-1]+scaled_grads[1:]),2)
    avg_scaled_grads=np.average(scaled_grads,axis=0)
    if graph == False:
        avg_scaled_grads_x=[avg_scaled_grads[tok] for tok in inp[1][0]]
        avg_scaled_grads_x=torch.Tensor(avg_scaled_grads_x).to(device)
    else:
        avg_scaled_grads_x = inp[0].x
        avg_scaled_grads_x = avg_scaled_grads_x.type(torch.FloatTensor)
        for j in range(len(avg_scaled_grads_x)):
            for jj in range(len(avg_scaled_grads_x[j])):
                avg_scaled_grads_x[j][jj] = avg_scaled_grads[int(avg_scaled_grads_x[j][jj])]
        avg_scaled_grads_x = avg_scaled_grads_x.to(device)
        
    integrated_gradients = (graph_or_token-baseline)*avg_scaled_grads_x
    integrated_gradients = rescale(np.array(integrated_gradients.to('cpu')))
    
    if graph == False:
        return(integrated_gradients[0])
    else:
        return(integrated_gradients)

def process_activation(sensitivity_weight, sensitivity_grad,ig,all_codes,encoder,all_predicted,all_lbl,all_filenames, *args):
    
    ## Optional if no argmax done yet
    #all_predicted = getClass(all_predicted)
    
    sensitivity_grad = rescale(sensitivity_grad)
    all_tokens, activations_weights, activations_grads=get_tokens_grads(all_codes,
                                                                      sensitivity_weight,
                                                                      sensitivity_grad,
                                                                      encoder)
    
    if len(args)==0:
        mydf = create_html(all_tokens,
                           all_predicted,
                           activations_weights,
                           activations_grads,
                           ig,
                           all_lbl,
                           all_filenames)
    else:
        mydf = create_html(all_tokens,
                           all_predicted,
                           activations_weights,
                           activations_grads,
                           ig,
                           all_lbl,
                           all_filenames,
                           args[0])

    return(mydf)


def display_weights(mydf,idx):
    plt.figure(figsize=(15, 18))
    plt.barh(mydf.function[idx], mydf.weights[idx], color='red')
    
def display_activations(mydf,idx,figsize=(10,12)):
    display(HTML(mydf.html_grads[idx]))
    display(HTML(mydf.html_ig[idx]))
    ax = pd.DataFrame({'Saliency maps': mydf.gradients[idx], 'Integrated grads': mydf.ig[idx]}, index=mydf.function[idx]).drop_duplicates().plot.barh(figsize=figsize,width=0.8)
    ax.set_xlabel("Activations")
    ax.set_ylabel("Source code tokens")
    
def display_func(mydf,idx):
    with open(Path(mydf.filename[idx]),'r') as k:
        print(k.read())