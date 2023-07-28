from utils.train_test import train, test

def run(model, device, trainloader, testloader, optimizer, criterion, epochs=20):
	for epoch in range(epochs):
		print("EPOCH:", epoch+1,'LR:',optimizer.param_groups[0]['lr'])
		train_loss, train_acc = train(model, device, trainloader, optimizer, criterion, epoch)
		test_loss , test_acc = test(model, device, criterion, testloader)
		scheduler.step(test_loss[-1])
	return train_acc, train_loss, test_acc, test_loss