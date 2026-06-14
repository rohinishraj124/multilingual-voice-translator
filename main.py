from models.model_manager import ModelManager
from gui.main_window import MainWindow


def main():
    print("\nLoading models, please wait...\n")
    manager = ModelManager()
    manager.load_models()
    app = MainWindow(manager)
    app.run()


if __name__ == "__main__":
    main()
