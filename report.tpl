<!DOCTYPE html>
<html lang="en">
<head>
  <title>CVX VS PP</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
</head>
<body>

<div class="container">
  <h1>CVX Vs. Proximal Point Comparison Report</h1>
  <p>This report presents a comparison between the CVX and the Proximal Point algorithms over different problems.</p>
  <p>In the table below the used Proximal Point parameters are displayed:</p>

  % if params is not None:
  <table class="table table-striped">
    <thead class="text-center bg-info">
      <th>Name</th>
      <th>Value</th>
    </thead>
    <tbody class="text-center">
      % for key, value in params.items():
      <tr>
        <td>${key}</td>
        <td>${value}</td>
      </tr>
      % endfor
    </tbody>
  </table>
  % endif

  <p>
    The table below represents an overview of how the 2 algorithms run. You can
    see that there are multiple comparison points that can be taken into consideration.
  </p>

  <table class="table table-striped">
    <thead class="bg-info text-center">
      <tr>
        <th>#</th>
        <th>Problem Name</th>
        <th>Algorithm Name</th>
        <th>Iters.</th>
        <th>Min. Value</th>
        <th>Time (s)</th>
      </tr>
    </thead>
    <tbody class="text-center">
      % for idx, problem in enumerate(problems):
      <% rowspan = len(problem['algorithms']) %>
      <tr>
        <td bgcolor="#bad6d3" rowspan="${rowspan}"><strong>${str(idx+1)}</strong></td>
        <td bgcolor="#bad6d3" rowspan="${rowspan}"><strong>${problem['name']}</strong></td>
        % for i, algo in enumerate(problem['algorithms']):
        % if i > 0:
          <tr>
        % endif
        <td>${str(algo['name'].upper())}</td>
        <td>${str(algo['iters'])}</td>
        <td>${str(algo['f_val'])}</td>
        <td>${str(algo['cpu_time'])}</td>
      </tr>
      % endfor
      % endfor
    </tbody>
  </table>
</div>

</body>
</html>
