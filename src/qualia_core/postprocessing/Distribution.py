from .PostProcessing import PostProcessing

import math
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import os
import re
from os import listdir
from os.path import isfile, join
import numpy as np

class Distribution(PostProcessing):
    def __init__(self, model: dict=None, method: str="layer_wise", bins: int=10, min: float=-5., max: float=5., output_layer:bool=True, output_pdf:bool=False):
        """ Generate a PDF with the distribution of all layers

            :param method: ["layer_wise", "network_wise"] Define the way to save the model parameters
            :param bins: Bins for histogram
            :param min: min for histogram border
            :param max: max for histogram border
            :param output_layer: generate .pkl with layer informations inside. Required for the first use.
            :param output_pdf: generate a pdf with distribution per {use method}
            :return trainresult:
        """
        print(f"{method=}")
        self.params  = {}
        self.bins = bins
        self.min    = min
        self.max    = max
        self.output_layer   = output_layer
        self.output_pdf     = output_pdf

        self.output_path    = Path('out')/'distribution'
        self.model_name     = ""

        if method in ["layer_wise", "network_wise"]:
            self.method = method
        else:
            print(f"INVALID METHOD USED :{method}: METHOD IS NOW SET TO 'layer_wise'")
            self.method = "layer_wise"

    def __call__(self, trainresult, model_conf):
        self.model      = trainresult.model
        self.model_name = trainresult.name
        if self.model is None :
            raise NotImplementedError()
        else:
            self.param_dict()
        
        if self.method == "layer_wise":
            self.distrib_layerwise()
        elif self.method == "network_wise":
            raise NotImplementedError()
            # self.distrib_networkwise() # [Deprecated]
        else:
            print(f"Method ERROR : {self.method=}. Must be layer_wise or network_wize")
            
        return trainresult, model_conf
        
    
    def param_dict(self):
        """
        build a dictionary of parameters dictionary sorted by layer, exemple:

            :return: {'weight'   : { 'layer1':weight_tensor1,'layer2':weight_tensor2 },

                     {'bias'     : { 'layer1':bias_tensor1, 'layer2':bias_tensor2 },

                     {'Other'    : { 'layer3':weight_tensor3}}
        """
        for name, param in self.model.named_parameters():
            layer_name, param_name = name.split(".")[0:-1],name.split(".")[-1]
            print(f"{name=} {layer_name=} {param_name=}")
            
            key_list = list(self.params.keys())
            if param_name in key_list:
                self.params[param_name]['.'.join(layer_name)]= param
            else:
                self.params[param_name] = {'.'.join(layer_name): param}
    
    def distrib_networkwise(self):
        """ Deprecated
        """
        pass
    
    def distrib_layerwise(self):
        """ Generate distribution savefiles that could be read to generate the PDF
        
            WIP - Working only with models parameters (not with activation)
        """
        # Generate files to save layers distribution informations 
        # Path : out/distribution/{network_name}
        output_model_dict   = dict()
        for param_name in list(self.params.keys()):
            layer_dict          = self.params[param_name]

            for layer_name in list(layer_dict.keys()):
                vector1d = torch.flatten(layer_dict[layer_name])
                container1d = vector1d

                min, max    = self.min_max(container1d)

                hist = torch.histc(container1d, bins = self.bins, min=min, max=max)   
                # remove all '.' in layer_name
                layer_name = layer_name.replace('.','')
                # if the layer name is ending with a number, rewrite it in format 00
                matches = re.findall(r"(\D+)(\d+)", layer_name)

                if matches:
                    layer_name = ""
                    for match in matches:
                        layer_name = f"{layer_name}{match[0]}{int(match[1]):02d}" 
                        

                print(f"{layer_name}:{param_name} min:{container1d.min().item()} - max:{container1d.max().item()} - mean:{container1d.mean().item()} - std:{container1d.std().item()}")

                output_model_dict[f"{layer_name}-{param_name}"] =[container1d.min().item(),container1d.max().item(),
                                                            container1d.mean().item(),container1d.std().item(),hist]
        
            # End for layers
        # end for model
        if self.output_layer:
            path = Path(f'{self.output_path}')/f'{self.model_name}'
            if not os.path.exists(path):
                os.makedirs(path)

            path = Path(f'{path}')/f'{self.model_name}_distribution.pkl'
            print(f"Saving in {path} ------------------------")
            f = open(path,"wb")
            pickle.dump(output_model_dict,f)
            f.close()
            
        if self.output_pdf:
            # Work only for generated file layers
            # must be rework to work "inline" without gerating files for distribution layer
            self.pdf_layerwise(line_per_page=4)

    def min_max(self, tens):
        """ Get the min and max of a tensor.
            
            Used for the histogram generated
            
            :params tens: 'class' torch tensor
            :return: min [int]

                     max [int]             
        """

        min_min = self.min
        max_max = self.max
        print(f"{tens.min()=} - {tens.max()=}")

        min = min_min if tens.min() < min_min else tens.min()
        max = max_max if tens.max() > max_max else tens.max()

        min, max = int(math.floor(min)), int(math.ceil(max))
        if min > max:
            tmp = max
            max = min
            min = tmp

        if min == max:
            min = min - 1
            max = max + 1 
        return min, max
    
    def pdf_layerwise(self, line_per_page:int=4):
        """ Generate pdf
        """
        
        network_dict = {}

        # mypath = join(self.output_path, self.model_name)
        mypath = join(self.output_path, self.model_name)
        print(f"{mypath=} - {self.output_path=} - {self.model_name=}")
        
        pkl_path = join(mypath, f"{self.model_name}_distribution.pkl")
        # get the model pkl file
        f = open(pkl_path,'rb')
        model_pkl = pickle.load(f)
        f.close()
        # get all layers names
        layers_pkl = list(model_pkl.keys())

        params_keys = []

        for layer_pkl in layers_pkl:
            # get network information with layers information stored
            print(f"{layer_pkl=}")
            layer_name, param_name = layer_pkl.split('-')
            
            min, max, mean, std, hist = model_pkl[f"{layer_name}-{param_name}"]
            
            print(f"{layer_name}")

            # Get param_type names (weight, bias, ...)
            if param_name not in params_keys:
                params_keys.append(param_name)

            # building network dict
            if layer_name in list(network_dict.keys()):
                network_dict[layer_name][param_name] = [min, max, mean, std, hist]
            else:
                network_dict[layer_name] = {param_name : [min, max, mean, std, hist]}

        keys = list(network_dict.keys())
        # sort by alphabetic order nb: work for conv and fully connected c < f 
        keys.sort()
        print(f"Layer names : {keys}")

        # option for subplot - sorted by layer type then parameters
        pdf_path = join(mypath, f"{self.model_name}-Distribution.pdf")
        pdf_pages = PdfPages(pdf_path)
        total_of_plots = len(keys) * len(params_keys)

        nb_of_pages = int(np.ceil(len(keys)/line_per_page))

        divided_keys = divide_list(keys, nb_of_pages, line_per_page)[0]
        print(f"{divided_keys=}")
        print(f"{divided_keys[0]=}")

        for page_nb, sub_keys in enumerate(divided_keys):
            fig, axes = plt.subplots(nrows=line_per_page, ncols=len(params_keys), figsize=(12, 8))
            for l_idx, l_key in enumerate(sub_keys):

                for p_idx, p_key in enumerate(params_keys):
                    # if the histogram exist
                    try:
                        # get the histogram
                        min, max, mean, std, hist = network_dict[sub_keys[l_idx]][params_keys[p_idx]]
                        dmin, dmax    = self.min_max(torch.tensor((min,max)))
                        print(f"{dmin=} - {dmax=}")
                        x = torch.arange(start=dmin,end=dmax,step=(dmax-dmin)/len(hist)).numpy()
                        x = x[0:len(hist)]
                        print(f"{sub_keys[l_idx]}:{params_keys[p_idx]} min:{min} - max:{max} - mean:{mean} - std:{std} - dmin:{dmin} - dmax:{dmax}")
                        # Create a histogram in the current subplot
                        axes[l_idx, p_idx].bar(x, hist, align='center', width=(max-min)/self.bins, linewidth=0)
                        axes[l_idx, p_idx].set_title(f"{l_key}_{p_key}|{min:4.2}|{max:4.2}|{mean:4.2}|{std:4.2}")
                    except KeyError:
                        print(f"No {params_keys[p_idx]} in {sub_keys[l_idx]} /!\\")
                        
            # Adjust spacing between subplots
            plt.tight_layout()
            # Add the current figure to the PDF
            pdf_pages.savefig(fig)

        # Close the PDF file
        pdf_pages.close()
        print(f"pdf saved : {self.output_path}")

def divide_list(input_list, m, c):
    """ Divide a list in m sublist of c element
    
    :params m: Number of sublist
    :params c: number of element per sublist

    :return: a list of all sublist od c element
    """
    # Check if n is divisible by both m and c
    if len(input_list) % (m * c) != 0:
        raise ValueError("Input list length is not divisible by m * c")

    # Divide the input list into m sub-lists of c elements each
    divided_lists = [input_list[i:i + c] for i in range(0, len(input_list), c)]

    # Split the divided_lists into m sub-lists
    final_lists = [divided_lists[i:i + m] for i in range(0, len(divided_lists), m)]

    return final_lists


