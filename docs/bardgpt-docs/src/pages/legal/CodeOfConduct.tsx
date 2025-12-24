import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";

export default function CodeOfConduct() {
  return (
    <DocLayout>
      <Breadcrumb items={[{ label: "Legal" }, { label: "Code of Conduct" }]} />
      <div className="doc-prose">
        <h1>Code of Conduct</h1>
        <p className="text-lg text-muted-foreground">Our commitment to a welcoming community.</p>
        <h2>Our Pledge</h2>
        <p>We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.</p>
        <h2>Our Standards</h2>
        <p>Examples of behavior that contributes to a positive environment:</p>
        <ul><li>Using welcoming and inclusive language</li><li>Being respectful of differing viewpoints</li><li>Gracefully accepting constructive criticism</li><li>Focusing on what is best for the community</li></ul>
        <h2>Enforcement</h2>
        <p>Project maintainers are responsible for clarifying standards of acceptable behavior and may take appropriate action in response to any behavior deemed inappropriate, threatening, or harmful.</p>
      </div>
    </DocLayout>
  );
}
