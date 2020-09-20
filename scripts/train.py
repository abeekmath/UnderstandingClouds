import torch


def train(dataloader_train, dataloader_test, model, epochs, loss_fn, optimizer):
    model.train()
    loss_epoch_arr = []
    min_loss = 1000
    n_iters = np.ceil(dataloader_train/batch_size)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if min_loss > loss.item():
                min_loss = loss.item()
                best_model = copy.deepcopy(model.state_dict())
                print('Min loss %0.2f' % min_loss)

            if i % 100 == 0:
                print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))
            
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        loss_epoch_arr.append(loss.item())
        
        print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
            epoch, epochs, 
            evaluation(dataloader_test, model), evaluation(dataloader_train, model)))

    return best_model