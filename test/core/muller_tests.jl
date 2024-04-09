@testitem "Muller" begin
    @testset "Quadratic function" begin
        f(u, p) = u^2 - p

        u0 = (10, 20, 30)
        p = 612
        prob = NonlinearProblem{false}(f, u0, p)
        sol = solve(prob, Muller())

        @test sol.u ≈ √612

        u0 = (-10, -20, -30)
        prob = NonlinearProblem{false}(f, u0, p)
        sol = solve(prob, Muller())

        @test sol.u ≈ -√612
    end

    @testset "Sine function" begin
        f(u, p) = sin(u)

        u0 = (1, 2, 3)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, Muller())

        @test sol.u ≈ π

        u0 = (2, 4, 6)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, Muller())

        @test sol.u ≈ 2*π
    end

    @testset "Exponential-sine function" begin
        f(u, p) = exp(-u)*sin(u)

        u0 = (-2, -3, -4)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, Muller())

        @test sol.u ≈ -π

        u0 = (-1, 0, 1/2)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, Muller())

        @test sol.u ≈ 0

        u0 = (-1, 0, 1)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, Muller())

        @test sol.u ≈ π
    end
end
