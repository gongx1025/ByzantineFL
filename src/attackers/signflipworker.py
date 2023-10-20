from src.worker import ByzantineWorker


class SignflippingWorker(ByzantineWorker):

    def get_gradient(self):
        return -super().get_gradient()