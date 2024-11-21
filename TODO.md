# TODO

- Rename ShapePoseSet obj file
- Implement dataloader for pytorch
  - Given random intance shape index i from 0 to 108 (length of beta)
  - Given random pose index j from pose list
- Load corresponding tshirt.obj
- Return corresponding sequences

- Chunkify dataloader, feed data to MLP + GRU + GNN network
- Optimize loss function
  - L2 norm
  - Normal smooth item
  - Laplacian smooth item
  - Bending loss (optional)
