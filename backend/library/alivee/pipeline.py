from .ecosystem import Ecosystem

class EntityBuilder:

    def __init__(self, pipeline, name):
        self.pipeline = pipeline
        self.name = name
        self.name = name
        self.inputs = []
        self.outputs = []
        self.config = None

    def configure(self, **kwargs):
        from .config import AgentConfig
        if self.config is None:
            self.config = AgentConfig()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
            else:
                print(f"Warning: Invalid config key '{k}'")
        return self

    def senses(self, valid_inputs):
        if isinstance(valid_inputs, str):
            valid_inputs = [valid_inputs]
        self.inputs.extend(valid_inputs)
        return self

    def expresses(self, valid_outputs):
        if isinstance(valid_outputs, str):
            valid_outputs = [valid_outputs]
        self.outputs.extend(valid_outputs)
        return self

    def build(self):
        return self.pipeline._register_entity(self)

class Pipeline:

    def __init__(self, use_sys_sensors=True):
        self.ecosystem = Ecosystem(use_sys_sensors=use_sys_sensors)
        self.builders = []

    def create_entity(self, name):
        builder = EntityBuilder(self, name)
        self.builders.append(builder)
        return builder

    def remove_entity(self, name):
        self.ecosystem.remove_agent(name)

    def freeze(self, name):
        agent = self.ecosystem.get_agent(name)
        if agent:
            agent.freeze()

    def unfreeze(self, name):
        agent = self.ecosystem.get_agent(name)
        if agent:
            agent.unfreeze()

    def pretrain(self, name, dataset, epochs=100):
        agent = self.ecosystem.get_agent(name)
        if agent:
            agent.pretrain(dataset, epochs=epochs)

    def restore(self, name, source_name=None):
        agent = self.ecosystem.get_agent(name)
        if agent:
            agent.load('./save_state', source_name=source_name)

    def _register_entity(self, builder):
        self.ecosystem.create_agent(name=builder.name, custom_inputs=builder.inputs, output_labels=builder.outputs, config=builder.config)
        return self

    def update(self, inputs={}):
        for builder in self.builders:
            if builder.name not in self.ecosystem.agents:
                self.ecosystem.create_agent(name=builder.name, custom_inputs=builder.inputs, output_labels=builder.outputs)
        return self.ecosystem.update(inputs)