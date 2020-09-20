import torch
import config 


def train_fn(model, data_loader, optimizer, critertion, epoch):
    model.train()
    fin_loss = 0
    for batch_id, data in enumerate(data_loader):
        inputs, targets = data["images"], data["targets"]
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

        optimizer.zero_grad()
        output = model(input)
        loss =  critertion(output, targets)                    
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_id * len(inputs), len(data_loader.dataset),
                100. * batch_id / len(data_loader), loss.data.item()))
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    test_loss = 0 
    correct = 0

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader): 
            batch_loss = 0
            inputs,  = data["images"], data["targets"]
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            _, outputs = model(inputs)
            batch_loss = loss_fn(outputs, targets)
            test_loss += batch_loss.item()

            pred = torch.round(outputs)
            correct += (pred == targets.data).sum().item()

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100.0 * correct / len(data_loader.dataset)

    return test_loss, test_accuracy