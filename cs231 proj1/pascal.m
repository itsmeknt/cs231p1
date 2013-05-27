function [ap] = pascal(cls, n)

  t =tic;
% [ap1, ap2] = pascal(cls, n)
% Train and score a model with n components.

globals;
pascal_init;

model = pascal_train(cls, n);
model.interval = 10;
boxes = pascal_test(cls, model, 'test', VOCyear);
ap = pascal_eval(cls, boxes, 'test', VOCyear);
model.interval = 5;
TIMME = toc(t)
