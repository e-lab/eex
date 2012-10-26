function edgeDetector(input)

	if input:size(1) ~= 1 then print("Error. Input must be grayscale") end
	
	ker = torch.Tensor(3,3):fill(-1)
	ker[2][2] = 8

	output = image.convolve(input,ker)
	
	return(output)
end
