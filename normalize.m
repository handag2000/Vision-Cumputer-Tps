function normalizedImage=normalize(image)

minVal=min(image(:));
maxVal=max(image(:));


normalizedImage=(image-minVal)/(maxVal-minVal)*255;


end

