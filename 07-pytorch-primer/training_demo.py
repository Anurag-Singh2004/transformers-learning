import torch
import torch.nn as nn

#Define a model
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x= self.linear1(x)
        x= self.relu(x)
        x= self.linear2(x)
        return x


#setup
model = FeedForward(d_model=4, d_ff=16)

x = torch.tensor([[1.1, 0.2, 0.7, 0.3]])
target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#Training loop
for step in range(200):
    output = model(x)
    loss = ((output-target)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")


print("\nFinal output:", output)
print("Target was: ", target)
