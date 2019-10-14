% Helper to calculate jacobian for p1

%% Define Transformation Matrices
syms L0 L1
% L0 = 0.1;
% L1 = 0.11;
% DH parameters
syms q0 q1
dh_parameters = [0, 0, q0, 0;
                 0, L0, q1, 0;
                 0, L1, 0, 0]

% calculate DH matrix
dh_matrix_list = sym(zeros(4, 4, 3))
for i = 1:3
    dh_matrix_list(:,:,i) = getDhMatrix(dh_parameters(i, 1), dh_parameters(i, 2), dh_parameters(i, 3), dh_parameters(i, 4))
end

H0_3 = dh_matrix_list(:, :, 1)*dh_matrix_list(:, :, 2)*dh_matrix_list(:, :, 3)

% to latex
disp(latex(H0_3))

% test
test_res = subs(subs(H0_3, q0, 0), q1, 0)   % q0 = 0, q1 = 0

%% Calculate Jacobian
f1 = H0_3(1, 4)
f2 = H0_3(2, 4)
f3 = q0 + q1;

jacobian_matrix = jacobian([f1, f2, f3], [q0, q1])
% to latex
disp(latex(jacobian_matrix))

%% Get traj x_dot, y_dot
syms theta
x = (0.19 + 0.02 * cos(4 * theta)) * cos(theta)
y = (0.19 + 0.02 * cos(4 * theta)) * sin(theta)
x_dot = diff(x, theta)
y_dot = diff(y, theta)

%% Helper functions
function[dh_matrix] = getDhMatrix(alpha, a, theta, d)
    dh_matrix = [cos(theta), -sin(theta), 0, a;
                 sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d;
                 sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d;
                 0, 0, 0, 1];
end
