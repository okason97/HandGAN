import torch
import torch.nn as nn

def test(image_datasets, dataloaders, model, device = 'cuda'):

    dataset_sizes = {'test': len(image_datasets['test'])}

    ###########
    #  TEST  #
    ###########
    print("Testing started")
    print('##############################')

    results = {
        'test': {
            'loss': None,
            'acc': None,
        },
    }

    criterion = nn.CrossEntropyLoss()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)                    
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    results['test']['loss'] = running_loss / dataset_sizes['test']
    results['test']['acc'] = running_corrects.double() / dataset_sizes['test']

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'Test', results['test']['loss'], results['test']['acc']))

    return results