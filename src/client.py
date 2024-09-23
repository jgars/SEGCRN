import click

from src.actions import start_evaluation, start_fidelity, start_training


@click.group("al_engine")
def client():
    pass


@client.command("train", short_help="Train the model on the indicated dataset.")
@click.option("--dataset", default="PEMSD4", help="Sets the dataset to train on.")
@click.option("--save-model", default=True, help="Sets whether the weights of the model are saved.")
def train_command(dataset, save_model):
    start_training(dataset, save_model=save_model)


@client.command("evaluate", short_help="Evaluate the model on the indicated dataset.")
@click.option("--dataset", default="PEMSD4", help="Sets the dataset to evaluate on.")
def evaluate_command(dataset):
    start_evaluation(dataset)


@client.command("fidelity", short_help="Calculte the fidelity and infidelity of the model on the indicated dataset.")
@click.option("--dataset", default="PEMSD4", help="Sets the dataset to evaluate on.")
def fidelity_command(dataset):
    start_fidelity(dataset)


if __name__ == "__main__":
    client()
