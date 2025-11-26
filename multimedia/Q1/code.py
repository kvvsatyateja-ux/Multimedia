from PIL import Image


#  DESATURATION GRAYSCALE
def desaturate_grayscale(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    gray_img = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            gray = (max(r, g, b) + min(r, g, b)) // 2
            gray_img.putpixel((x, y), (gray, gray, gray))


    gray_img.save(output_path)
    print("Saved:", output_path)

# 2) MEDIAN CUT QUANTIZATION
def median_cut_quantize(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    quantized_img = img.quantize(colors=64, method=0)
    quantized_img = quantized_img.convert("RGB")
    quantized_img.save(output_path)
    print("Saved:", output_path)
 

# 3) OCTREE QUANTIZATION
def octree_quantize(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    quantized_img = img.quantize(colors=64, method=2)
    quantized_img.save(output_path)
    print("Saved:", output_path)


# MAIN
input_file = "input.jpg"  
desaturate_grayscale(input_file, "output_desaturate.jpg")
median_cut_quantize(input_file, "output_mediancut.jpg")
octree_quantize(input_file, "output_octree.png")

print("All three outputs generated successfully!")
