def updateweight(model,LR):
    for param in model.parameters():
        param.data = param.data-param.grad*LR
    model.zero_grad()