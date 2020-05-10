""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, num_actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, num_actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        unnormalized_logits = self.fc(cat_in)
        action = torch.argmax(unnormalized_logits, dim=1).unsqueeze(dim=1)
        return action

