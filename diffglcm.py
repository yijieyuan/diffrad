import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum

class DiffGLCM(nn.Module):
    """
    Differentiable Grey Level Co-occurrence Matrix (GLCM) computation with feature extraction.
    
    This module calculates GLCMs from input images with support for differentiable operations,
    making it suitable for integration in neural network training pipelines.
    
    GLCMs are texture descriptors that capture spatial relationships between pixels by counting
    how often pairs of pixels with specific values occur at a specified displacement.
    
    Args:
        image_size (int, optional): Size of the input images (H=W). Defaults to 64.
        low_bound (float, optional): Lower bound of pixel values. Defaults to 0.
        high_bound (float, optional): Upper bound of pixel values. Defaults to 1.
        Ng (int, optional): Number of grey levels in the GLCM. Defaults to 64. Note: the discretization is left <= x < right.
        alpha (float, optional): Steepness parameter for sigmoid function when using 
                                 differentiable mode. Defaults to 10.
        differentiable (bool, optional): Whether to use differentiable operations.
                                        Defaults to True.
    """
    def __init__(self, image_size=64, low_bound=0, high_bound=1, Ng=64, alpha=10, differentiable=True):
        super(DiffGLCM, self).__init__()
        self.image_size = image_size
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.Ng = Ng
        self.alpha = alpha
        self.differentiable = differentiable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Register the shift vector as a buffer so it's properly moved to the right device
        self.register_buffer('shift_vector', torch.arange(1, Ng+1, dtype=torch.float))
        
        # Initialize feature computation components
        self.NgVector = np.arange(1, self.Ng + 1, dtype="float32")
        self.i, self.j = np.meshgrid(self.NgVector, self.NgVector, indexing='ij', sparse=True)
        self.register_buffer('ng_vector', torch.from_numpy(self.NgVector).float())
        self.register_buffer('i_matrix', torch.from_numpy(self.i).float())
        self.register_buffer('j_matrix', torch.from_numpy(self.j).float())
        self.register_buffer('k_value_sum', torch.arange(2, (Ng * 2) + 1).float())  # shape = (2*Ng-1)
        self.register_buffer('k_values_diff', torch.arange(0, Ng).float())  # shape = (Ng-1)

    def compute_glcm(self, x, offset_r, offset_c):
        """
        Compute the GLCM for the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, H, W].
            offset_r (int): Row offset for co-occurrence calculation
            offset_c (int): Column offset for co-occurrence calculation.
            
        Returns:
            torch.Tensor: Normalized GLCM of shape [batch_size, 1, Ng, Ng, 1].
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Normalize input to [1, Ng+1] range
        x_normalized = (x - self.low_bound) / (self.high_bound - self.low_bound) * self.Ng + 1
        
        # Reshape shift vector for broadcasting with x
        shift_vector = self.shift_vector.view(1, -1, 1, 1)
        
        # Repeat x along channel dimension for vectorized comparison
        x_repeated = x_normalized.repeat(1, self.Ng, 1, 1)
        
        # Compute accumulation mask to determine bin membership
        accum_mask = x_repeated - shift_vector
        
        # Convert to binary mask using either hard threshold or sigmoid
        if not self.differentiable:
            accum_mask = (accum_mask >= 0).float()
        else:
            accum_mask = torch.sigmoid(self.alpha * accum_mask)
        
        # Create auxiliary matrices for proper bin handling
        ones_matrix = torch.ones((batch_size, 1, self.image_size, self.image_size), 
                                device=device, dtype=torch.float)
        zeros_matrix = torch.zeros((batch_size, 1, self.image_size, self.image_size), 
                                 device=device, dtype=torch.float)
        
        # Handle boundary cases
        accum_mask = torch.cat([ones_matrix, accum_mask[:, 1:], zeros_matrix], dim=1)

        # Compute bin membership by taking differences between consecutive accum_masks
        threshold_mask = accum_mask[:, :-1] - accum_mask[:, 1:]

        # Extract paired regions based on specified row-and-column offsets
        if offset_c == 0 and offset_r == 0:  
            raise ValueError('offset_r and offset_c cannot be both 0')
        
        elif offset_c >= 0:
            if offset_r == 0:  # Only horizontal offset
                center_region = threshold_mask[:, :, :, :-offset_c]  # Left portion
                peripheral_region = threshold_mask[:, :, :, offset_c:]  # Right portion
                
            elif offset_c == 0:  # Only vertical offset
                center_region = threshold_mask[:, :, :-offset_r, :]  # Top portion
                peripheral_region = threshold_mask[:, :, offset_r:, :]  # Bottom portion
                
            elif offset_c > 0 and offset_r > 0:  # Both horizontal and vertical offset
                center_region = threshold_mask[:, :, :-offset_r, :-offset_c]  # Top-left portion
                peripheral_region = threshold_mask[:, :, offset_r:, offset_c:]  # Bottom-right portion

        # Handle negative column offset:
        elif offset_c < 0:
            if offset_r == 0:  # Only horizontal offset (negative)
                center_region = threshold_mask[:, :, :, -offset_c:]  # Right portion
                peripheral_region = threshold_mask[:, :, :, :offset_c]  # Left portion
                
            elif offset_r > 0:  # Vertical and horizontal (negative) offset
                center_region = threshold_mask[:, :, :-offset_r, -offset_c:]  # Top-right portion
                peripheral_region = threshold_mask[:, :, offset_r:, :offset_c]  # Bottom-left portion

        # Reshape the regions for further processing
        threshold_masks_center = rearrange(center_region, 'batch bin h w -> batch bin (h w)')
        threshold_masks_peripheral = rearrange(peripheral_region, 'batch bin h w -> batch (h w) bin')
        
        # Compute co-occurrence matrix via matrix multiplication (einsum)
        glcm = einsum(threshold_masks_center, threshold_masks_peripheral, 
                      'batch bin1 n, batch n bin2 -> batch bin1 bin2').unsqueeze(1).float()
        
        # Normalize GLCM to create a probability distribution
        glcm = glcm / torch.sum(glcm, dim=(1, 2, 3), keepdim=True)
        
        # Reshape to [batch_size, 1, Ng, Ng, 1] to match feature computer expected format
        glcm = glcm.unsqueeze(-1)
        
        return glcm

    def compute_features(self, glcm):
        """
        Compute texture features from the GLCM.
        
        Args:
            glcm (torch.Tensor): GLCM tensor of shape [batch_size, 1, Ng, Ng, Na].
            
        Returns:
            torch.Tensor: GLCM features of shape [batch_size, 24, Na].
        """
        # Prepare variables
        device = glcm.device
        Ng = self.Ng
        i = self.i_matrix
        j = self.j_matrix
        kValuesSum = self.k_value_sum
        kValuesDiff = self.k_values_diff
        batch_size = glcm.shape[0]
        Na = glcm.shape[-1]
        
        # Symmetrize the GLCM
        P_glcm = glcm
        # P_glcm = glcm + torch.transpose(glcm, 2, 3).clone()
        # P_glcm = P_glcm / torch.sum(P_glcm, dim=(2, 3), keepdim=True) 

        # Small value to avoid log(0)
        eps = 1e-10
        
        # Calculate marginal probabilities
        px = torch.sum(P_glcm, dim=3, keepdim=True)  # shape = (batch_size, 1, Ng, 1, Na)
        py = torch.sum(P_glcm, dim=2, keepdim=True)  # shape = (batch_size, 1, 1, Ng, Na)

        # Calculate means
        ux = torch.sum(i[None, None, :, :, None] * P_glcm, dim=(2, 3), keepdim=True)
        uy = torch.sum(j[None, None, :, :, None] * P_glcm, dim=(2, 3), keepdim=True)

        # Calculate p(x+y) and p(x-y)
        pxAddy = torch.stack([torch.sum(P_glcm[:, :, i + j == k, :], dim=2) for k in kValuesSum]).permute(1, 2, 0, 3)
        pxSuby = torch.stack([torch.sum(P_glcm[:, :, torch.abs(i - j) == k, :], dim=2) for k in kValuesDiff]).permute(1, 2, 0, 3)

        # Joint entropy
        HXY = (-1) * torch.sum(P_glcm * torch.log2(P_glcm + eps), dim=(2, 3))

        # Feature 1 - Autocorrelation
        autocorrelation = torch.sum(P_glcm * (i * j)[None, None, :, :, None], dim=(2, 3)).squeeze()
        
        # Feature 2 - Joint Average
        joint_average = ux.squeeze()
        
        # Feature 3 - Cluster Prominence
        cluster_prominence = torch.sum((P_glcm * ((i + j)[None, None, :, :, None] - ux - uy) ** 4), dim=(2, 3)).squeeze()
        
        # Feature 4 - Cluster Shade
        cluster_shade = torch.sum((P_glcm * ((i + j)[None, None, :, :, None] - ux - uy) ** 3), dim=(2, 3)).squeeze()
        
        # Feature 5 - Cluster Tendency
        cluster_tendency = torch.sum((P_glcm * ((i + j)[None, None, :, :, None] - ux - uy) ** 2), dim=(2, 3)).squeeze()
        
        # Feature 6 - Contrast
        contrast = torch.sum(P_glcm * (i - j)[None, None, :, :, None] ** 2, dim=(2, 3)).squeeze()
        
        # Feature 7 - Correlation
        sigx = torch.sum(P_glcm * (i[None, None, :, :, None] - ux) ** 2, dim=(2, 3), keepdim=True) ** 0.5
        sigy = torch.sum(P_glcm * (j[None, None, :, :, None] - uy) ** 2, dim=(2, 3), keepdim=True) ** 0.5
        corm = torch.sum(P_glcm * (i[None, None, :, :, None] - ux) * (j[None, None, :, :, None] - uy), dim=(2, 3), keepdim=True)
        corr = corm / (sigx * sigy + eps)
        correlation = corr.squeeze()

        # Feature 8 - Difference Average
        difference_average = torch.sum((kValuesDiff[None, None, :, None] * pxSuby), dim=(2)).squeeze()

        # Feature 9 - Difference Entropy
        difent = (-1) * torch.sum(pxSuby * torch.log2(pxSuby + eps), dim=(2))
        difference_entropy = difent.squeeze()

        # Feature 10 - Difference Variance
        diffavg = torch.sum((kValuesDiff[None, None, :, None] * pxSuby), dim=(2), keepdim=True)
        diffvar = torch.sum((pxSuby * (kValuesDiff[None, None, :, None] - diffavg) ** 2), dim=(2))
        difference_variance = diffvar.squeeze()

        # Feature 11 - Joint Energy
        joint_energy = torch.sum(P_glcm ** 2, dim=(2, 3)).squeeze()

        # Feature 12 - Joint Entropy
        joint_entropy = HXY.squeeze()

        # Feature 13 - Informational Measure of Correlation 1
        HX = (-1) * torch.sum(px * torch.log2(px + eps), dim=(2,3))
        HY = (-1) * torch.sum(py * torch.log2(py + eps), dim=(2,3))
        HXY1 = (-1) * torch.sum(P_glcm * torch.log2(px * py + eps), dim=(2,3))
        div = torch.fmax(HX, HY)
        imc1 = HXY - HXY1
        imc1 = torch.where(div != 0, imc1 / div, torch.zeros_like(imc1))
        imc1 = imc1.squeeze()

        # Feature 14 - Informational Measure of Correlation 2
        HXY2 = (-1) * torch.sum((px * py) * torch.log2(px * py + eps), dim=(2, 3))
        imc2 = (1 - torch.exp(-2 * (HXY2 - HXY))) ** 0.5
        imc2 = torch.where(HXY2 == HXY, torch.zeros_like(imc2), imc2)
        imc2 = imc2.squeeze()

        # Feature 15 - Inverse Difference Moment 
        idm = torch.sum((pxSuby / (1 + (kValuesDiff[None, None, :, None] ** 2))), dim=(2)).squeeze()

        # Feature 16 - Maximum Correlation Coefficient
        if not self.differentiable:
            P_glcmLeft = P_glcm.permute(0, 1, 2, 3, 4) # shape = (batch_size, 1, Ng, Ng, 4)
            P_glcmRight = P_glcm.permute(0, 1, 3, 2, 4) # shape = (batch_size, 1, Ng, Ng, 4)
            # P_norm_x = px.permute(0, 1, 3, 2, 4).repeat(1, 1, Ng, 1, 1)
            # P_norm_y = py.permute(0, 1, 3, 2, 4).repeat(1, 1, 1, Ng, 1)

            P_norm_x = px.repeat(1, 1, 1, Ng, 1)
            P_norm_y = py.repeat(1, 1, Ng, 1, 1)

            mat1 = rearrange(P_glcmLeft / (P_norm_x + eps), '... i j k -> ... k i j')  # Swap axes 2 and 3
            mat2 = rearrange(P_glcmRight / (P_norm_y + eps), '... j i k -> ... k j i')  # Swap axes 2 and 3
            Q = mat1 @ mat2
            Q = rearrange(Q, '... k i j -> ... i j k')  # Swap axes 1 and 2
            Q_eigenValues = torch.linalg.eigvalsh(Q.permute(0, 1, 4, 2, 3))
            second_largest_Q_eigenValue = Q_eigenValues[..., -2]
            mcc = torch.sqrt(second_largest_Q_eigenValue.real).squeeze()
        else:
            mcc = torch.zeros_like(idm, device=device)

        # Feature 17 - Inverse Difference Moment Normalized
        idmn = torch.sum((pxSuby / (1 + (((kValuesDiff[None, None, :, None] ** 2) / (Ng ** 2))))), dim=(2)).squeeze()

        # Feature 18 - Inverse Difference
        id_feature = torch.sum(pxSuby / (1 + kValuesDiff[None, None, :, None]), dim=(2)).squeeze()

        # Feature 19 - Inverse Difference Normalized
        idn = torch.sum(pxSuby / (1 + (kValuesDiff[None, None, :, None] / Ng)), dim=(2)).squeeze()

        # Feature 20 - Inverse Variance
        inverse_variance = torch.sum(pxSuby[:, :, 1:, :] / (kValuesDiff[None, None, 1:, None] ** 2), dim=(2)).squeeze()

        # Feature 21 - Maximum Probability
        maximum_probability = torch.amax(P_glcm, dim=(2,3)).squeeze()

        # Feature 22 - Sum Average
        sum_average = torch.sum(kValuesSum[None, None, :, None] * pxAddy, dim=(2)).squeeze()

        # Feature 23 - Sum Entropy
        sum_entropy = (-1) * torch.sum(pxAddy * torch.log2(pxAddy + eps), dim=(2)).squeeze()

        # Feature 24 - Sum of Squares: Variance
        sum_squares = torch.sum(P_glcm * ((i[None, None, :, :, None] - ux) ** 2), dim=(2, 3)).squeeze()
        
        # Reshape all features to consistent dimensions
        autocorrelation = autocorrelation.view(batch_size, Na)
        cluster_prominence = cluster_prominence.view(batch_size, Na)
        cluster_shade = cluster_shade.view(batch_size, Na)
        cluster_tendency = cluster_tendency.view(batch_size, Na)
        contrast = contrast.view(batch_size, Na)
        correlation = correlation.view(batch_size, Na)
        difference_average = difference_average.view(batch_size, Na)
        difference_entropy = difference_entropy.view(batch_size, Na)
        difference_variance = difference_variance.view(batch_size, Na)
        id_feature = id_feature.view(batch_size, Na)
        idm = idm.view(batch_size, Na)
        idmn = idmn.view(batch_size, Na)
        idn = idn.view(batch_size, Na)
        imc1 = imc1.view(batch_size, Na)
        imc2 = imc2.view(batch_size, Na)
        inverse_variance = inverse_variance.view(batch_size, Na)
        joint_average = joint_average.view(batch_size, Na)
        joint_energy = joint_energy.view(batch_size, Na)
        joint_entropy = joint_entropy.view(batch_size, Na)
        mcc = mcc.view(batch_size, Na)
        maximum_probability = maximum_probability.view(batch_size, Na)
        sum_average = sum_average.view(batch_size, Na)
        sum_entropy = sum_entropy.view(batch_size, Na)
        sum_squares = sum_squares.view(batch_size, Na)

        # Stack all features together
        feature_output = torch.stack([
            autocorrelation,         # 1
            cluster_prominence,      # 3
            cluster_shade,           # 4
            cluster_tendency,        # 5
            contrast,                # 6
            correlation,             # 7
            difference_average,      # 8
            difference_entropy,      # 9
            difference_variance,     # 10
            id_feature,              # 18
            idm,                     # 15
            idmn,                    # 17  
            idn,                     # 19
            imc1,                    # 13
            imc2,                    # 14
            inverse_variance,        # 20
            joint_average,           # 2
            joint_energy,            # 11
            joint_entropy,           # 12
            mcc,                     # 16
            maximum_probability,     # 21
            sum_average,             # 22
            sum_entropy,             # 23
            sum_squares              # 24
        ], dim=1)

        return feature_output.float()  # shape = (batch_size, 24, Na)

    def forward(self, x, offset_r, offset_c):
        """
        Complete forward pass to compute GLCM and extract features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, H, W].
            offset_r (int): Row offset for co-occurrence calculation
            offset_c (int): Column offset for co-occurrence calculation.
            
        Returns:
            tuple: (glcm, features)
                - glcm: GLCM tensor of shape [batch_size, 1, Ng, Ng, 1]
                - features: GLCM features of shape [batch_size, 24, 1]
        """
        # Compute GLCM
        glcm = self.compute_glcm(x, offset_r, offset_c)
        
        # Compute features from GLCM
        features = self.compute_features(glcm)
        
        return glcm, features