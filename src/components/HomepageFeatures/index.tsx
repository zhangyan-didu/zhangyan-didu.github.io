import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Tech Blog',
    Svg: require('@site/static/img/techblog.svg').default,
    description: (
      <>
        AI / AI infra / Basics
      </>
    ),
  },
  {
    title: 'Feeling',
    Svg: require('@site/static/img/feelings.svg').default,
    description: (
      <>
        留住瞬间思绪
      </>
    ),
  },
  {
    title: 'Project',
    Svg: require('@site/static/img/projects.svg').default,
    description: (
      <>
        个人项目展示（包括施工中的项目）
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  const getLinkUrl = (title: string) => {
    switch(title) {
      case 'Tech Blog':
        return '/blog';
      case 'Feeling':
        return '/feelings';
      case 'Project':
        return '/projects';
      default:
        return '/docs/intro';
    }
  };

  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to={getLinkUrl(title)}
          >
            Show me
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
