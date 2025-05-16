import Header from "../common/Header";
import { MaxWidthWrapper } from "@/components/common/max-width-wrapper"
import Image from "next/image"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import "./ServiceSection.css"

function ServiceSection() {

  const codeSnippet = `class GNN(torch.nn.Module):
    def __init__(self, metadata, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv2 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv3 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv4 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean') 
        self.conv5 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean') 
        self.lin = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv5(x_dict, edge_index_dict)

        return self.lin(x_dict["transaction"]).squeeze(-1)`

return(

<div className="s-section">
<section className="relative py-24 sm:py-32 bg-brand-25">
<MaxWidthWrapper className="flex flex-col items-center gap-16 sm:gap-20">
<Header title="service" subtitle="Our Vision & Our Goal" />

  <div className="grid gap-4 lg:grid-cols-3 lg:grid-rows-2">

    

    <div className="relative lg:row-span-2">
      <div className="absolute inset-px rounded-lg bg-white lg:rounded-l-[2rem]" />

      <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] lg:rounded-l-[calc(2rem+1px)]">
        <div className="px-8 pb-3 pt-8 sm:px-10 sm:pb-0 sm:pt-10">
          <p className="mt-2 text-lg/7 font-medium tracking-tight text-brand-950 max-lg:text-center">
          Risk Scoring & Analysis
          </p>
          <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
          Automatically assess transaction risks with advanced AI-driven risk scoring to proactively mitigate potential fraud threats.
          </p>
        </div>

        <div className="relative min-h-[30rem] w-full grow [container-type:inline-size] max-lg:mx-auto max-lg:max-w-sm">
          <div className="absolute inset-x-10 bottom-0 top-10 overflow-hidden  shadow-2xl">
            <Image
              className="size-full object-cover object-top"
              src="/images/risk_analysis_icon.png"
              alt="Phone screen displaying app interface"
              fill
            />
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 lg:rounded-l-[2rem]" />

    </div> 



    
    <div className="relative max-lg:row-start-1">
      <div className="absolute inset-px rounded-lg bg-white max-lg:rounded-t-[2rem]" />
      <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] max-lg:rounded-t-[calc(2rem+1px)]">
        <div className="px-8 pt-8 sm:px-10 sm:pt-10">
          <p className="mt-2 text-lg/7 font-medium tracking-tight text-brand-950 max-lg:text-center">
          Behavioral Analysis
          </p>
          <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
          Analyze user behaviors and payment patterns to differentiate between legitimate users and potential fraudsters.
          </p>
        </div>
        <div className="flex flex-1 items-center justify-center px-8 max-lg:pb-12 max-lg:pt-10 sm:px-10 lg:pb-2">
          <Image
            className="w-full max-lg:max-w-xs"
            src="/images/behavior_analysis_icon.png"
            alt="Bento box illustrating event tracking"
            width={500}
            height={300}
          />
        </div>
      </div>

      <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 max-lg:rounded-t-[2rem]" />

    </div>

   



    <div className="relative max-lg:row-start-3 lg:col-start-2 lg:row-start-2">
      <div className="absolute inset-px rounded-lg bg-white" />
      <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)]">
        <div className="px-8 pt-8 sm:px-10 sm:pt-10">
          <p className="mt-2 text-lg/7 font-medium tracking-tight text-brand-950 max-lg:text-center">
            Data Visualization & Insights
          </p>
          <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
           Get comprehensive dashboards and visual insights into fraud trends, transaction patterns, and security metrics.
          </p>
        </div>

        <div className="flex flex-1 items-center justify-center px-8 max-lg:pb-12 max-lg:pt-10 sm:px-10 lg:pb-2">
          <Image
            className="w-full max-lg:max-w-xs"
            src="/images/data_visualization_icon.png"
            alt="Bento box illustrating custom data tracking"
            width={500}
            height={300}
          />
        </div>
      </div>

      <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5" />
    </div>

    

    
    <div className="relative lg:row-span-2">
      <div className="absolute inset-px rounded-lg bg-white max-lg:rounded-b-[2rem] lg:rounded-r-[2rem]" />

      <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] max-lg:rounded-b-[calc(2rem+1px)] lg:rounded-r-[calc(2rem+1px)]">
        <div className="px-8 pb-3 pt-8 sm:px-10 sm:pb-0 sm:pt-10">
          <p className="mt-2 text-lg/7 font-medium tracking-tight text-brand-950 max-lg:text-center">
            Easy Integration
          </p>
          <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
          Seamlessly integrate fraud detection capabilities into your existing systems with simple APIs and flexible deployment options.
          </p>
        </div>

        <div className="relative min-h-[30rem] w-full grow">
          <div className="absolute bottom-0 left-10 right-0 top-10 overflow-hidden rounded-tl-xl bg-gray-900 shadow-2xl">
            <div className="flex bg-gray-800/40 ring-1 ring-white/5">
              <div className="-mb-px flex text-sm/6 font-medium text-gray-400">
                <div className="border-b border-r border-b-white/20 border-r-white/10 bg-white/5 px-4 py-2 text-white">
                  graphguard.py
                </div>
              </div>
            </div>

            <div className="overflow-hidden">
              <div className="max-h-[30rem]">

                <SyntaxHighlighter
                  language="typescript"
                  style={{
                    ...oneDark,
                    'pre[class*="language-"]': {
                      ...oneDark['pre[class*="language-"]'],
                      background: "transparent",
                      overflow: "hidden",
                    },
                    'code[class*="language-"]': {
                      ...oneDark['code[class*="language-"]'],
                      background: "transparent",
                    },
                  }}
                >
                  {codeSnippet}
                </SyntaxHighlighter>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 max-lg:rounded-b-[2rem] lg:rounded-r-[2rem]" />
    </div>
  </div>
</MaxWidthWrapper>
</section>
</div>

)
}

export default ServiceSection;