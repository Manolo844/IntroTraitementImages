import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_points.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        img = mpimg.imread(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    print(f"Image loaded: {image_path}")
    print("Cliquer 4 points sur l'image.")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Cliquer 4 points. Les coordonnées seront affichées après 4 clics.")
    
    # ginput(4) will wait for 4 clicks
    # timeout=0 means no timeout
    points = plt.ginput(4, timeout=0)
    plt.close(fig)

    if len(points) < 4:
        print("Required 4 clicks, but only got", len(points))
        sys.exit(1)

    # points is a list of (x, y) tuples
    x_coords = [int(p[0]) for p in points]
    y_coords = [int(p[1]) for p in points]

    print("\nCoordonnées à copier-coller:")
    print(f"x = {x_coords}")
    print(f"y = {y_coords}")

if __name__ == "__main__":
    main()
