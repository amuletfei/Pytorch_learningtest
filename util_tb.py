from tensorboardX import SummaryWriter
import numpy as np




# log_dir=dir of tensorboardfiles foldername='logs'
writer = SummaryWriter(log_dir='logs', flush_secs=60)


#create graph,model=pytorch model,input_to_model =pytorch model input
if Cuda:
    graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],imput_shape[1])).type(torch.FloatTensor).cuda()
else
    graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],imput_shape[1])).type(torch.FloatTensor)
writer.add_graph(model(graph_inputs,))


#draw loss function
writer.add_scalar('Train_loss', loss, (epoch*epoch_size+iteration))


#view in cmd:tensorboard --logdir=url,url is logs's dir