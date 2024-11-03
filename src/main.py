import torch

def main():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    # Perform a simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y

    print(f"Result of tensor addition: {z}")

if __name__ == "__main__":
    main()
