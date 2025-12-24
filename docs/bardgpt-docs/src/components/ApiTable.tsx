interface Parameter {
  name: string;
  type: string;
  description: string;
  default?: string;
}

interface ApiTableProps {
  parameters: Parameter[];
  title?: string;
}

export function ApiTable({ parameters, title = "Parameters" }: ApiTableProps) {
  return (
    <div className="mb-6">
      <h4 className="text-sm font-medium text-muted-foreground mb-3">{title}</h4>
      <div className="overflow-x-auto border border-border rounded-lg">
        <table className="api-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Description</th>
              <th>Default</th>
            </tr>
          </thead>
          <tbody>
            {parameters.map((param) => (
              <tr key={param.name}>
                <td>
                  <code className="text-primary text-sm">{param.name}</code>
                </td>
                <td>
                  <code className="text-muted-foreground text-xs">{param.type}</code>
                </td>
                <td className="text-sm">{param.description}</td>
                <td>
                  {param.default ? (
                    <code className="text-xs text-muted-foreground">{param.default}</code>
                  ) : (
                    <span className="text-muted-foreground/50 text-xs">Required</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
