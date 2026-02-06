import psutil
from .agent import Agent
from .config import AgentConfig

class Ecosystem:

    def __init__(self, use_sys_sensors=True):
        self.agents = {}
        self.shared_ether = {}
        self.use_sys_sensors = use_sys_sensors

    def create_agent(self, name, output_labels, custom_inputs=[], config=None):
        std_inputs = ['sys_cpu', 'sys_ram', 'sys_pwr'] if self.use_sys_sensors else []
        all_inputs = std_inputs + custom_inputs
        new_agent = Agent(name, all_inputs, output_labels, config)
        self.agents[name] = new_agent
        return new_agent

    def get_agent(self, name):
        return self.agents.get(name)

    def remove_agent(self, name):
        if name in self.agents:
            del self.agents[name]
        if name in self.shared_ether:
            del self.shared_ether[name]

    def _read_hardware(self):
        cpu = psutil.cpu_percent() / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        try:
            batt = psutil.sensors_battery()
            pwr = batt.percent / 100.0 if batt else 1.0
            charging = batt.power_plugged if batt else True
        except:
            pwr, charging = (1.0, True)
        critical = pwr < 0.2 and (not charging)
        self.shared_ether['sys_cpu'] = cpu
        self.shared_ether['sys_ram'] = ram
        self.shared_ether['sys_pwr'] = pwr
        return (charging, critical)

    def update(self, external_data={}):
        charging, critical = (False, False)
        if self.use_sys_sensors:
            charging, critical = self._read_hardware()
        self.shared_ether.update(external_data)
        world_snapshot = {}
        for name, agent in self.agents.items():
            input_vec = []
            for key in agent.input_keys:
                val = self.shared_ether.get(key, 0.0)
                input_vec.append(val)
            target = external_data.get(f'{name}_target')
            outputs = agent.think(input_vec, charging, critical, target=target)
            for key, val in outputs.items():
                self.shared_ether[f'{name}_{key}'] = val
            world_snapshot[name] = {'state': agent.current_state, 'data': outputs, 'loss': agent.memory_loss}
        return world_snapshot

    def save_snapshot(self, folder='./save_state'):
        for agent in self.agents.values():
            agent.save(folder)

    def load_snapshot(self, folder='./save_state'):
        for agent in self.agents.values():
            agent.load(folder)