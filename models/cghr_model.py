import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks


def register_custom_modules():
    from models.cbam import CBAM
    setattr(modules, 'CBAM', CBAM)
    setattr(tasks, 'CBAM', CBAM)
    print("[Course] CBAM registered in modules + tasks")
