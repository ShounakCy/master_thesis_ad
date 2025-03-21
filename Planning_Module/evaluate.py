import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import pipNet
from data import highwayTrajDataset
from utils import initLogging, maskedMSETest, maskedNLLTest
import matplotlib.pyplot as plt
import os
from pdb import set_trace as bp
import csv
## Network Arguments
parser = argparse.ArgumentParser(description='Evaluate :Trajectory Prediction for Autonomous Driving')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default = True)
parser.add_argument('--use_planning', action="store_false", help='if use planning  (default: True)', default = True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion (default: True)', default = True)
parser.add_argument('--batch_size', type=int, help='batch size (default: 64)', default=64)
parser.add_argument('--train_flag', action="store_true", help='if concatenate with true maneuver label (default: No)', default = False)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int,  help='default: (25,5)', nargs=2,       default = [25, 5])
parser.add_argument('--in_length', type=int,  help='history sequence (default: 16)', default = 16)
parser.add_argument('--out_length', type=int, help='predict sequence (default: 25)', default = 25)
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors',   default = 3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors', default = 2)
# Network hyperparameters------------------------------------------
parser.add_argument('--temporal_embedding_size', type=int,  help='Embedding size of the input traj', default = 32)
parser.add_argument('--encoder_size', type=int, help='lstm enc size',  default = 64)
parser.add_argument('--decoder_size', type=int, help='lstm dec size',  default = 128)
parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth', default = 16)
parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic', default = 32)
parser.add_argument('--social_context_size', type=int,  help='Embedding size of social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',   default = 112)
## Evaluation setting------------------------------------------
parser.add_argument('--name',     type=str, help='model name', default="1")
parser.add_argument('--test_set', type=str, help='Path to test datasets', default="/efs/workspace/PiP-Planning-informed-Prediction/datasets/test.mat")
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--metric',   type=str, help='RMSE & NLL is calculated ', default="agent")
parser.add_argument("--plan_info_ds", type=int, default=1, help="N, further downsampling planning information to N*0.2s")


def model_evaluate():

    args = parser.parse_args()

    ## Initialize network
    PiP = pipNet(args)
    args.name = "ngsim_model" #highD_model
    args.test_set = "/efs/workspace/PiP-Planning-informed-Prediction/datasets/test.mat"
    PiP.load_state_dict(torch.load('./trained_models/{}/{}.tar'.format((args.name).split('-')[0], args.name)))
    if args.use_cuda:
        PiP = PiP.cuda()

    ## Evaluation Mode
    PiP.eval()
    PiP.train_flag = False
    initLogging(log_file='./trained_models/{}/evaluation.log'.format((args.name).split('-')[0]))

    ## Intialize dataset
    logging.info("Loading test data from {}...".format(args.test_set))
    tsSet = highwayTrajDataset(path=args.test_set,
                               targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                               grid_size=args.grid_size,
                               fit_plan_traj=True,
                               fit_plan_further_ds=args.plan_info_ds)
    logging.info("TOTAL :: {} test data.".format(len(tsSet)) )
    tsDataloader = DataLoader(tsSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tsSet.collate_fn)
    
    ## Loss statistic
    logging.info("<{}> evaluated by {}-based NLL & RMSE, with planning input of {}s step.".format(args.name, args.metric, args.plan_info_ds*0.2))
    if args.metric == 'agent':
        nll_loss_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                  np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        rmse_loss_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                   np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        both_count_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                    np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
    
    else:
        raise RuntimeError("Wrong type of evaluation metric is specified")
    avg_eva_time = 0

    ## Evaluation process
    with torch.no_grad():
        for i, data in enumerate(tsDataloader):
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idxs = data
            # Initialize Variables
            print(i)
            if args.use_cuda:
                nbsHist = nbsHist.cuda()
                nbsMask = nbsMask.cuda()
                planFut = planFut.cuda()
                planMask = planMask.cuda()
                targsHist = targsHist.cuda()
                targsEncMask = targsEncMask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                targsFut = targsFut.cuda()
                targsFutMask = targsFutMask.cuda()

            # Inference
            st_time_inf = time.time()
            fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            sp_time = time.time() - st_time
            print("inf_time : ",sp_time)
            import pandas as pd
            targsFut_y = targsFut[:,:,0].detach().cpu().numpy() #convert to Numpy array
            targsFut_x = targsFut[:,:,1].detach().cpu().numpy()
            targsHist_y = targsHist[:,:,0].detach().cpu().numpy()
            targsHist_x = targsHist[:,:,1].detach().cpu().numpy()
            planFut_y = planFut[:,:,0].detach().cpu().numpy()
            planFut_x = planFut[:,:,1].detach().cpu().numpy()
            
            df_targsFut_y = pd.DataFrame(targsFut_y) #convert to a dataframe
            df_targsFut_x = pd.DataFrame(targsFut_x) #convert to a dataframe
            df_targsHist_y = pd.DataFrame(targsHist_y) #convert to a dataframe
            df_targsHist_x = pd.DataFrame(targsHist_x) #convert to a dataframe
            df_planFut_y = pd.DataFrame(planFut_y) #convert to a dataframe
            df_planFut_x = pd.DataFrame(planFut_x) #convert to a dataframe

            df_targsFut_y.to_csv("df_targsFut_y",index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe
            df_targsFut_x.to_csv("df_targsFut_x",index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe
            df_targsHist_y.to_csv("df_targsHist_y",index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe
            df_targsHist_x.to_csv("df_targsHist_x",index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe
            df_planFut_y.to_csv("df_planFut_y",index=False) #save to file 
            df_planFut_x.to_csv("df_planFut_x",index=False) #save to file 

            for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    fut_pred_i_y = fut_pred_i[:,:,0].numpy() #convert to Numpy array
                    fut_pred_i_x = fut_pred_i[:,:,1].numpy()
                    df_fut_pred_i_y = pd.DataFrame(fut_pred_i_y) #convert to a dataframe
                    df_fut_pred_i_x = pd.DataFrame(fut_pred_i_x) #convert to a dataframe
                    df_fut_pred_i_y.to_csv("df_fut_pred_i_y_"+str(j),index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe
                    df_fut_pred_i_x.to_csv("df_fut_pred_i_x_"+str(j),index=False) #save to file 'df = pd.DataFrame(t_np) #convert to a dataframe

            # Performance metric
            #bp()
            indices = []
            if args.metric == 'agent':
                dsIDs, targsIDs = tsSet.batchTargetVehsInfo(idxs)
                
                # Select the trajectory with the largest probability of maneuver label when evaluating by RMSE
                if  not args.train_flag:
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man * 3 + lat_man
                        indices.append(indx)

                        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                    # Using the most probable trajectory
                    ll, cc = maskedMSETest(fut_pred_max, targsFut, targsFutMask, separately=True)
                    l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, separately=True)
                    l = l.detach().cpu().numpy()
                    ll = ll.detach().cpu().numpy()
                    c = c.detach().cpu().numpy()
                    cc = cc.detach().cpu().numpy()
                    lat_man = torch.argmax(lat_enc, dim=-1).detach()
                    lon_man = torch.argmax(lon_enc, dim=-1).detach()
                    #train_flag = args.train_output_flag
                    import pandas as pd
                    t_np = targsFut[:,:,0].detach().cpu().numpy() #convert to Numpy array
                    df = pd.DataFrame(t_np) #convert to a dataframe
                    df.to_csv("targsFut",index=False) #save to file 
                    #bp()
                    lat_man = torch.argmax(lat_enc, dim=-1).detach()
                    lon_man = torch.argmax(lon_enc, dim=-1).detach()
                    draw(targsHist, targsFut, nbsHist, nbsMask, fut_pred, args.train_flag, lon_man, lat_man, indices, planFut)
                else:
                    #ll, cc = maskedMSETest(fut_pred, targsFut, targsFutMask, separately=True)
                    #l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, separately=True)
                    #l = l.detach().cpu().numpy()
                    #ll = ll.detach().cpu().numpy()
                    #c = c.detach().cpu().numpy()
                    #cc = cc.detach().cpu().numpy()
                    lat_man = torch.argmax(lat_enc, dim=-1).detach()
                    lon_man = torch.argmax(lon_enc, dim=-1).detach()
                    #train_flag = args.train_output_flag
                    import pandas as pd
                    t_np = targsFut[:,:,0].detach().cpu().numpy() #convert to Numpy array
                    df = pd.DataFrame(t_np) #convert to a dataframe
                    df.to_csv("targsFut",index=False) #save to file 
                    #bp()
                    lat_man = torch.argmax(lat_enc, dim=-1).detach()
                    lon_man = torch.argmax(lon_enc, dim=-1).detach()
                    draw(targsHist, targsFut, nbsHist, nbsMask, fut_pred, args.train_flag, lon_man, lat_man, None, planFut)

                #for j, targ in enumerate(targsIDs):
                #    dsID = dsIDs[j]
                #    nll_loss_stat[dsID, targ, :]   += l[:, j]
                #    rmse_loss_stat[dsID, targ, :]  += ll[:, j]
                #    both_count_stat[dsID, targ, :]  += c[:, j]
                #bp()
            

            # Time estimate
            batch_time = time.time() - st_time
            avg_eva_time += batch_time
            if i%100 == 99:
                eta = avg_eva_time / 100 * (len(tsSet) / args.batch_size - i)
                logging.info( "Evaluation progress(%):{:.2f}".format( i/(len(tsSet)/args.batch_size) * 100,) +
                              " | ETA(s):{}".format(int(eta)))
                avg_eva_time = 0

    # Result Summary
    if args.metric == 'agent':
        # Loss averaged from all predicted vehicles.
        ds_ids, veh_ids = both_count_stat[:,:,0].nonzero()
        num_vehs = len(veh_ids)
        rmse_loss_averaged = np.zeros((args.out_length, num_vehs))
        nll_loss_averaged = np.zeros((args.out_length, num_vehs))
        count_averaged = np.zeros((args.out_length, num_vehs))
        for i in range(num_vehs):
            count_averaged[:, i] = \
                both_count_stat[ds_ids[i], veh_ids[i], :].astype(bool)
            rmse_loss_averaged[:,i] = rmse_loss_stat[ds_ids[i], veh_ids[i], :] \
                                      * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
            nll_loss_averaged[:,i]  = nll_loss_stat[ds_ids[i], veh_ids[i], :] \
                                      * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
        rmse_loss_sum = np.sum(rmse_loss_averaged, axis=1)
        nll_loss_sum = np.sum(nll_loss_averaged, axis=1)
        count_sum = np.sum(count_averaged, axis=1)
        rmseOverall = np.power(rmse_loss_sum / count_sum, 0.5) * 0.3048  # Unit converted from feet to meter.
        nllOverall = nll_loss_sum / count_sum
    

    # Print the metrics every 5 time frame (1s)
    logging.info("RMSE (m)\t=> {}, Mean={:.3f}".format(rmseOverall[4::5], rmseOverall[4::5].mean()))
    logging.info("NLL (nats)\t=> {}, Mean={:.3f}".format(nllOverall[4::5], nllOverall[4::5].mean()))

def add_nhbr_car(plt, x, y, alp):
        plt.gca().add_patch(plt.Rectangle(
            (x , y - 2.5),  
            5,  
            2.5, 
            color='dimgrey',
            alpha=alp
        ))

def add_ego_car(plt, x, y, alp):
        plt.gca().add_patch(plt.Rectangle(
            (x, y - 2.5),  
            5,  
            2.5,
            color='maroon',
            alpha=alp
        ))

def add_target_car(plt, x, y, alp):
        plt.gca().add_patch(plt.Rectangle(
            (x , y - 2.5),  
            5,  
            2.5,
            color='darkgreen',
            alpha=alp
        ))
                

#   draw(targsHist, targsFut, nbsHist, nbsMask, targs_fut_pred, train_flag, lon_man, lat_man, indices, planFut)
def draw(hist, fut, nbrs, mask, fut_pred, train_flag,lon_man, lat_man, indices, planFut):

        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        planFut = planFut.cpu()
        #fut_pred = fut_pred.cpu()
        
        
        IPL = 0
        scale = 0.3048
        prop =1
        op =0
        for i in range(hist.size(1)):
            lon_man_i = lon_man[i].item()
            lat_man_i = lat_man[i].item()
            plt.axis('on')
            plt.ylim(-18 * scale, 18 * scale)
            plt.xlim(-180 * scale , 180 * scale )
            plt.figure(dpi=300)
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / 64).item())
            for ii in range(IPL_i):
                plt.plot(hist[:, IPL + ii, 1]*scale , hist[:, IPL + ii, 0]*scale , '-',
                         color='grey',
                         linewidth=0.5)
               

                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    plt.plot(fut_pred_i[:, i, 1]*scale , fut_pred_i[:, i, 0]*scale ,'-.', 
                        color='green', linewidth=0.5)
                
                
           
                add_target_car(plt, hist[-1, i, 1]*scale, hist[-1, i, 0]*scale, alp=0.5)
                plt.plot(fut[:, i, 1]*scale , fut[:, i, 0]*scale , '-', 
                    color='yellow',
                    linewidth=0.5)
            
                
                plt.plot(planFut[:, i, 1]*scale , planFut[:, i, 0]*scale , '-', 
                        color='red',
                        linewidth=0.5)
            
                add_ego_car(plt, planFut[-1, i, 1]*scale, planFut[-1, i, 0]*scale, alp=0.5)
            IPL = IPL + IPL_i         
            plt.gca().set_aspect('equal', adjustable='box')
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'highD_Results/')
            sample_file_dir = os.path.join(results_dir,str(lon_man_i + 1) + '_' + str(lat_man_i + 1)+'/')
            file = str(op)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            if not os.path.isdir(sample_file_dir):
                os.makedirs(sample_file_dir)
            plt.savefig(sample_file_dir+ file + '.png')
            op += 1
   
            plt.close()         


if __name__ == '__main__':
    model_evaluate()