import ast
import os
import importlib.util
from typing import Dict, List, Type, Any

def find_and_load_classes_inheriting_from_metricabc(folder_path: str, target_class: str = 'MetricABC') -> Dict[str, Type[Any]]:
    """
    Find and load all classes in the specified folder that inherit from the target_class.

    Args:
        folder_path (str): Path to the folder containing Python files.
        target_class (str): Name of the class to check inheritance from.

    Returns:
        Dict[str, Type[Any]]: A dictionary mapping class names to their class objects.
    """
    class_dict = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]

                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # Check all attributes of the module for classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        # Check if the class inherits from target_class
                        if any(base.__name__ == target_class for base in attr.__bases__):
                            class_dict[attr_name] = attr

    return class_dict

# Example usage:
folder_path = 'metrics'  # Replace with your folder path
class_dict = find_and_load_classes_inheriting_from_metricabc(folder_path)

# Now you can instantiate any of the classes:
for class_name, class_obj in class_dict.items():
    print(f"Found class: {class_name}")
    # Example instantiation (adjust arguments as needed)
    try:
        instance = class_obj()
        print(f"Successfully instantiated {class_name}")
    except Exception as e:
        print(f"Failed to instantiate {class_name}: {e}")