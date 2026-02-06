import torch
import torch.optim as optim
import os
import json
from .core import TriStateBrain
from .config import AgentConfig
import torch.nn.functional as F

class Agent:

    def __init__(self, name, input_keys, output_keys, config: AgentConfig=None):
        self.name = name
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.cfg = config if config else AgentConfig()
        self.cfg.input_dim = len(input_keys)
        self.cfg.output_dim = len(output_keys)
        self.brain = TriStateBrain(self.cfg)
        self.optimizer = optim.AdamW([p for p in self.brain.parameters() if p.requires_grad], lr=self.cfg.inner_lr)
        self.current_state = 'BOOT'
        self.memory_loss = 0.0
        self.is_frozen = False

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def think(self, input_vector, is_charging, is_critical, target=None):
        x = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        if is_critical:
            self.current_state = 'HIBERNATE'
            self.brain.eval()
            with torch.no_grad():
                out = self.brain(x)
            return self._format_output(out)
        if self.is_frozen:
            self.current_state = 'FROZEN'
            self.brain.eval()
            with torch.no_grad():
                out = self.brain(x)
            return self._format_output(out)
        self.current_state = 'ACTIVE'
        self.brain.train()
        out = self.brain(x)
        if target is not None:
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            if target_tensor.shape[-1] != out.shape[-1]:
                min_dim = min(target_tensor.shape[-1], out.shape[-1])
                loss = F.mse_loss(out[..., :min_dim], target_tensor[..., :min_dim])
            else:
                loss = F.mse_loss(out, target_tensor)
        else:
            target_energy = torch.mean(torch.abs(x))
            output_energy = torch.mean(out)
            loss = F.mse_loss(output_energy, target_energy)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory_loss = loss.item()
        if is_charging:
            self.current_state = 'CONSOLIDATING'
            self.current_state = 'CONSOLIDATING'

            def recursive_consolidate(module):
                if hasattr(module, 'consolidate_memory'):
                    module.consolidate_memory(rate=self.cfg.outer_lr)
                for child in module.children():
                    recursive_consolidate(child)
            recursive_consolidate(self.brain)
            self.optimizer = optim.AdamW([p for p in self.brain.parameters() if p.requires_grad], lr=self.cfg.inner_lr)
        return self._format_output(out)

    def _format_output(self, tensor_out):
        vals = tensor_out.detach().squeeze().tolist()
        if isinstance(vals, float):
            vals = [vals]
        return dict(zip(self.output_keys, vals))

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.brain.state_dict(), f'{folder}/{self.name}.pt')
        meta = {'name': self.name, 'inputs': self.input_keys, 'outputs': self.output_keys, 'config': self.cfg.__dict__}
        with open(f'{folder}/{self.name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=4)

    def load(self, folder, source_name=None):
        target_name = source_name if source_name else self.name
        try:
            self.brain.load_state_dict(torch.load(f'{folder}/{target_name}.pt'))
            print(f'[{self.name}] Brain loaded from {target_name}.pt')
        except FileNotFoundError:
            print(f'[{self.name}] No save found for {target_name}. Starting fresh.')

    def pretrain(self, dataset, epochs=100):
        print(f'[{self.name}] Pretraining on {len(dataset)} samples for {epochs} epochs...')
        self.brain.train()
        all_inputs = [d['inputs'] for d in dataset]
        all_targets = [d['outputs'] for d in dataset]
        inputs_t = torch.tensor(all_inputs, dtype=torch.float32)
        targets_t = torch.tensor(all_targets, dtype=torch.float32)
        batch_size = 32
        num_samples = len(dataset)
        for epoch in range(epochs):
            total_loss = 0
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_in = inputs_t[indices]
                batch_tgt = targets_t[indices]
                out = self.brain(batch_in)
                if batch_tgt.shape[-1] != out.shape[-1]:
                    min_dim = min(batch_tgt.shape[-1], out.shape[-1])
                    loss = F.mse_loss(out[..., :min_dim], batch_tgt[..., :min_dim])
                else:
                    loss = F.mse_loss(out, batch_tgt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(indices)
            avg_loss = total_loss / num_samples
            if epoch % 10 == 0:
                print(f'  Epoch {epoch}: Loss {avg_loss:.4f}')
        for module in self.brain.modules():
            if hasattr(module, 'consolidate_memory'):
                module.consolidate_memory(rate=1.0)
        self.optimizer = optim.AdamW([p for p in self.brain.parameters() if p.requires_grad], lr=self.cfg.inner_lr)