import time
import torch
from torch.cuda.amp import autocast,GradScaler 
from torch import Tensor 
from typing import Tuple,Union,Any,TypeVar
from torch.nn.modules.linear import Linear
import warnings
cpu_device = torch.device('cpu')

loss_func  = TypeVar('loss_func')
optim = TypeVar('optim')
function = TypeVar('function')

class Fit:
    metrics_threshold:float = 0.0

    @classmethod
    def Train(cls,
        model:torch.nn.Module,epochs:int,batch_size:int,
        criterion:Union[loss_func,Tuple[loss_func,...]],
        optimizer:optim,
        #optimizer:Union[optim,Tuple[optim,...]],
        train_x:Union[Tensor,Tuple[Tensor,...]],
        train_y:Union[Tensor,Tuple[Tensor,...]],
        device:torch.device,
        val_x:Union[Tensor,Tuple[Tensor,...]]=None,
        val_y:Union[Tensor,Tuple[Tensor,...]]=None,
        auto_casting:bool = True,
        validatioin_split:float = 0.0,
        scheduler:torch.optim.lr_scheduler=None,
        #scheduler:Union[torch.optim.lr_scheduler,Tuple[torch.optim.lr_scheduler,...]]=None,
        metrics:Union[function,Tuple[Any,...]] = None,
        x_preprocesser:Union[function,Tuple[function,...]] = None,
        y_preprocesser:Union[function,Tuple[function,...]] = None,
        ) -> Tuple[Any]:
        """
        Args:
            model [required]:your pytorch model.
            criterion [required]: loss functions. You can set multiple it with tuple for your model.
            optimizer [required]: optimizer class. 
            train_x [required]: training data. You can input multiple it with tuple for your model
            train_y [required]: training answer. You can input mutiple it with tuple for your model.
            device [required]: torch.device 
        """
        device = torch.device(device)
        
        ##  change to tuple
        _t = type(criterion)
        if not (_t is tuple or _t is list):
            criterion = (criterion,)

        _t = type(train_x)
        if not (_t is tuple or _t is list):
            train_x = (train_x,)

        _t = type(train_y)
        if not (_t is tuple or _t is list):
            train_y = (train_y,)
        
        _t = type(val_x)
        if not (_t is tuple or _t is list):
            val_x = (val_x,)
        
        _t = type(val_y)
        if not (_t is tuple or _t is list):
            val_y = (val_y,)

        _t = type(metrics)
        if not (_t is tuple or _t is list):
            metrics = (metrics,)
        
        _t = type(x_preprocesser)
        if not (_t is tuple or _t is list):
            x_preprocesser = (x_preprocesser,)
        
        _t = type(y_preprocesser)
        if not (_t is tuple or _t is list):
            y_preprocesser = (y_preprocesser,)

        # validation splitting
        train_len = len(train_x[0])
        val_len = 0
        if val_x[0] is None and val_y[0] is None:
            if validatioin_split > 0.0:
                vs = round(train_len * validatioin_split)
                ts = train_len - vs
                train_x = [i[:ts] for i in train_x]
                train_y = [i[:ts] for i in train_y]
                val_x = [i[-vs:] for i in train_x]
                val_y = [i[-vs:] for i in train_y]
                train_len = ts
                val_len = vs
        elif validatioin_split > 0.0:
            val_len = len(val_x[0])
            warnings.warn('Ignore validation_split because you entered validation data')
        
        # type checking
        _preprox = x_preprocesser[0] is not None
        _preproy = y_preprocesser[0] is not None 
        _metrics = metrics[0] is not None
        _sch = scheduler is not None
        _cpu = device == cpu_device
        _eva = val_x[0] is not None
        # matching the number of inputs and outputs

        arglen = len(train_x)
        flen = len(x_preprocesser)
        if flen != 1 and arglen != flen:#  and _preprox:
            raise Exception(f'The number of x_preprocesser does not match the number of x. x_preprocesser:{flen},x:{arglen}')
        x_preprocesser = x_preprocesser * arglen


        arglen = len(train_y)
        crilen = len(criterion)
        if crilen != 1 and arglen != crilen:
            raise Exception(f'The number of criterion does not match the number of y. criterion:{crilen}, y:{arglen}')
        criterion = criterion * arglen

        flen = len(y_preprocesser)
        if flen != 1 and arglen != flen:
            raise Exception(f'The number of y_preprocesser does not match the number of y. y_preprocesser:{flen},y:{arglen}')
        y_preprocesser = y_preprocesser * arglen

        flen = len(metrics)
        if flen != 1 and arglen != flen:
            raise Exception(f'The number of metrics does not match the number of y. metrics:{flen},y:{arglen}')
        metrics = metrics * arglen
        
        # set result
        losses = torch.zeros(epochs,arglen)
        val_losses = torch.zeros(epochs,arglen)
        met_results = torch.zeros(epochs,arglen)
        val_met_results = torch.zeros(epochs,arglen)


        model = model.to(device) # send model to device


        maxcount = (train_len -1 ) // batch_size + 1
        val_maxcount = (val_len - 1) // batch_size + 1
        scaler = GradScaler()
        for epoch in range(epochs):
            message = ''
            model.train()
            print(f'epoch:{epoch+1}')
            start = time.time()

            # set result
            avg_loss = torch.zeros(arglen)
            avg_val_loss = torch.zeros(arglen)
            avg_met_result = torch.zeros(arglen)
            avg_val_met_results = torch.zeros(arglen)


            for counter,data_idx in enumerate(range(0,train_len,batch_size),1):
                _mes = '\r'
                data_x = [i[data_idx:data_idx+batch_size].to(device) for i in train_x]
                data_y = [i[data_idx:data_idx+batch_size].to(device) for i in train_y]

                if _preprox:
                    data_x = [x_p(i) for i,x_p in zip(data_x,x_preprocesser)]
                if _preproy:
                    data_y = [y_p(i) for i,y_p in zip(data_y,y_preprocesser)]

                # data input
                optimizer.zero_grad()
                if _cpu:
                    output = model(*data_x)
                    if type(output) != tuple:
                        output = (output,)
                    loss = [cri(i,a) for (cri,i,a) in zip(criterion,output,data_y)]
                    for i in loss:
                        i.backward()
                    optimizer.step()
                else:
                    with autocast(auto_casting):
                        output = model(*data_x)
                        if type(output) != tuple:
                            output = [output]
                        loss = [cri(i,a) for (cri,i,a) in zip(criterion,output,data_y)]
                        for i in loss:
                            scaler.scale(i).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                per = counter / maxcount * 100
                _mes = _mes + '{:5.2f}% '.format(per)
                # result saving
                for i,l in enumerate(loss):
                    r = l.item()
                    avg_loss[i] += r 
                    _mes += 'loss {}:{:5.5f} '.format(i,r)
                    
                if _metrics:
                    mvalues = [m(o,a) for (m,o,a) in zip(metrics,output,data_y)]
                    for i,m in enumerate(mvalues):
                        avg_met_result[i] += m
                        _mes += 'metrics {}:{:5.5f} '.format(i,m)
                print(_mes,end='')
            print('')

            # result saving
            avg_loss /= maxcount
            for i,l in enumerate(avg_loss):
                message += 'loss {}:{:5.5f} '.format(i,l)
                losses[epoch] = l

            if _metrics:                  
                avg_met_result /= maxcount
                for i,m in enumerate(avg_met_result):
                    message += 'metrics {}:{:5.5f} '.format(i,m)
                    met_results[epoch] = m

            
            ## Evaluating

            if _eva:
                model.eval()
                with torch.no_grad():
                    for counter,data_idx in enumerate(range(0,val_len,batch_size),1):
                        _mes = '\r'
                        data_x = [i[data_idx:data_idx+batch_size].to(device) for i in val_x]
                        data_y = [i[data_idx:data_idx+batch_size].to(device) for i in val_y]

                        if _preprox:
                            data_x = [i(d) for (i,d) in zip(x_preprocesser,data_x)]
                        if _preproy:
                            data_y = [i(d) for (i,d) in zip(y_preprocesser,data_y)]

                        if _cpu:
                            output = model(*data_x)
                        else:
                            with autocast(auto_casting):
                                output = model(*data_x)
                        if type(output) != tuple:
                            output = [output]
                        loss = [cri(i,a) for (cri,i,a) in zip(criterion,output,data_y)]

                        # result saving
                        per = counter / val_maxcount * 100
                        _mes += '{:5.2f}% '.format(per)

                        for i,l in enumerate(loss):
                            r = l.item()
                            avg_val_loss[i] += r 
                            _mes += 'avg_loss {}:{:5.5f} '.format(i,r)

                        if _metrics:
                            mvalues = [m(o,a) for (m,o,a) in zip(metrics,output,data_y)]
                            for i,m in enumerate(mvalues):
                                avg_val_met_results[i] += m
                                _mes += 'avg_metrics {}:{:5.5f} '.format(i,m)
                        
                        print(_mes,end='')
                    print('')
                    
                    # result saving
                    avg_val_loss /= val_maxcount
                    for i,l in enumerate(avg_val_loss):
                        message += 'avg_val_loss {}:{:5.5f} '.format(i,l)
                        val_losses[epoch] = l 
                    if _metrics:
                        avg_val_met_results /= val_maxcount
                        for i,m in enumerate(avg_met_result):
                            message += 'avg_val_metrics {}:{:5.5f} '.format(i,m)
                            val_met_results[epoch] = m
            lre = optimizer.param_groups[0]['lr']
            message += 'lr :{:5.5f} '.format(lre)
            if _sch:
                scheduler.step()
            
            elapsed = time.time() - start
            printlen = len(message)
            print('-'*printlen)
            print(message)
            print('-'*printlen)
            tani = elapsed * (epochs - epoch -1)
            print(f'\n{int(tani//3600)} : {int((tani%3600//60))} : {str(tani%60)[:5]}\n')  
            print('='*printlen)
        print('finished')
        
        returns = (losses,)
        if _eva:
            returns += (val_losses,)
        if _metrics:
            returns += (met_results,)
        if _eva and _metrics:
            returns += (val_met_results,)
        
        return returns

    @classmethod
    def Predict(cls,
        model:torch.nn.Module,
        data:Union[Tensor,Tuple[Tensor,...]],
        batch_size:int,
        device:torch.device,
        preprocesser:Union[function,Tuple[function,...]] = None,
        ) -> torch.Tensor:

        model.eval()
        model = model.to(device)
        _t = type(data)
        if not(_t is tuple or _t is list):
            data = (data,)
        
        _t = type(preprocesser)
        if not(_t is tuple or _t is list):
            preprocesser = (preprocesser,)
        
        arglen = len(data)
        flen = len(preprocesser)
        if flen != 1 and arglen != flen:
            raise Exception(f'The number of preprocesser does not match the number of data. preprocesser:{flen},data:{arglen}')
        preprocesser = preprocesser * arglen

        _prepro = preprocesser[0] is not None
        datalen = len(data[0])

        outputs = []
        model = model.type(data.dtype)

        with torch.no_grad():
            for counter,data_idx in enumerate(range(0,datalen,batch_size),1):
                data = [i[data_idx:data_idx+batch_size].to(device) for i in data]
                if _prepro:
                    data = [f(i) for f,i in zip(preprocesser,data)]
                output = model(*data).to('cpu')
                outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs


    @classmethod 
    def Accuracy(cls,output:Tensor,answer:Tensor) -> Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape

        output[output >= cls.metrics_threshold] = 1
        output[output < cls.metrics_threshold] = 0

    #length = torch.prod(torch.tensor(output.shape))
        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

    #error = torch.div(torch.sum(output==answer),length)
        TP = torch.sum((output==True)==(answer==True))
        FP = torch.sum((output==True)==(answer==False))
        TN = torch.sum((output==False)==(answer==False))
        FN = torch.sum((output==False)==(answer==True))
        error = (TP+TN) / (TP+FP+FN+TN)
        return error
                    
    @classmethod
    def Precision(cls,output:Tensor,answer:Tensor) -> Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape

        output[output >= cls.metrics_threshold] = 1
        output[output < cls.metrics_threshold] = 0

        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

        TP = torch.sum((output==True)==(answer==True))
        FP = torch.sum((output==True)==(answer==False))
        error = TP/(TP+FP)
        return error

    @classmethod
    def Recall(cls,output:Tensor,answer:Tensor) -> Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape

        output[output >= cls.metrics_threshold] = 1
        output[output < cls.metrics_threshold] = 0

        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

        TP = torch.sum((output==True)==(answer==True))
        FN = torch.sum((output==False)==(answer==True))

        error = TP/(TP+FN)
        return error  

    @classmethod
    def Specificity(cls,output:Tensor,answer:Tensor) -> Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape
        output[output >= cls.metrics_threshold] = 1
        output[output < cls.metrics_threshold] = 0

        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

        TN = torch.sum((output==True)==(answer==False))
        FP = torch.sum((output==True)==(answer==False))
        error = TN/(FP+TN)
        return error

    @classmethod
    def F_measure(cls,output:Tensor,answer:Tensor) -> Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape

        ###error = (2*Precision(output,answer*Recall(output,answer))) / (Precision(output,answer)+Recall(output,answer))
        output[output >= cls.metrics_threshold] = 1
        output[output < cls.me] = 0

        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

        TP = torch.sum((output==True)==(answer==True))
        FP = torch.sum((output==True)==(answer==False))
        FN = torch.sum((output==False)==(answer==True))

        error = (2*TP**2)/ (TP*(2*TP+FN+FP))
        return error

if __name__ == '__main__':
    from torch.nn import Module,Linear
    class test(Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(100,10)
        
        def forward(self,x1,x2):
            x1 = self.layer(x1)
            x2 = self.layer(x2)
            return x1,x2
    train_x = torch.randn(64,100)
    train_y = torch.randn(64,10)
    
    model = test()
    epochs = 5
    batch_size = 4
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    device = 'cpu'
    x_preproc = lambda x:x
    y_preproc = lambda y:y

    def met(a,b):
        return torch.tensor(1)
    sch = lambda epoch: 0.80 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,sch)

    x = Fit.Train(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        criterion=[criterion,criterion],
        optimizer=optimizer,
        train_x=[train_x,train_x],
        train_y=[train_y,train_y],
        val_x=[train_x,train_x],
        val_y=[train_y,train_y],
        validatioin_split=0.1,
        x_preprocesser=x_preproc,
        y_preprocesser=y_preproc,
        metrics=met,
        device=device,
        scheduler=scheduler
        

    )
    print(x)
    print(len(x))
    Fit.metrics_threshold = 1.0
    print(Fit.metrics_threshold)

